from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json
from tqdm import tqdm
from argparse import ArgumentParser
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args, tokenizer, model, image_processor, context_len ):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name
    # )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs

start = """Please serve as an unbiased judge in assessing the quality of the responses from AI assistants regarding the user's instruction and a figure. """

settings = {"COT figure": """Please examine the provided image attentively. Begin by conducting a comprehensive analysis of the figure provided. Detail your observations and insights in the 'Figure Analysis' section. Next, utilize the insights from your initial analysis to critically evaluate the responses. Summarize this evaluation in the 'Analysis' section. Finally, based on your figure analysis and response evaluation, form a well-reasoned judgement. Document this in the 'Judgement' section. Ensure that your final output in a JSON format with keys: 'Figure Analysis' for the initial figure assessment, 'Analysis' for the evaluation of responses against your ground truth, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "COT instruction": """Please examine the provided image attentively. Begin by providing a detailed response to the user instructions, treating this response as the baseline or 'ground truth'. This response will form the 'Response' section. Next, use this established ground truth to systematically analyze and evaluate the responses to the same instruction. This evaluation will form the 'Analysis' section. After the analysis, move forward to the judgement phase, where you will give final judgement based on the analysis of the responses compared to the ground truth. Give your judgement in the 'Judgement' section. Ensure that your final output is structured in a JSON format with the keys 'Response' for the answer to the instruction, 'Analysis' for the evaluation of responses, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "COT figure instruction": """Please examine the provided image attentively. Begin with an in-depth analysis of the figure. Detail your observations and insights in the 'Figure Analysis' section. Then, provide a detailed response to the user instructions, treating this as 'Response' and the ground truth. Next, compare and analyze the responses to the same instruction against your ground truth in the 'Analysis' section. Finally, give your final judgement in 'Judgement'. Structure your output in JSON format, with the following keys: 'Figure Analysis' for the initial figure assessment, 'Response' for your response to the instructions, 'Analysis' for the evaluation of responses against your ground truth, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "No COT": """Please examine the provided image attentively. Output your judgement in a JSON format with the key 'Judgement' only.""",
            "No Figure": """As a blind judge, you will not have access to the figure mentioned in the user instructions. Your task is to impartially assess the responses based solely on the information presented within them, without visual context of the figure. Begin by performing a detailed analysis of the responses, capturing your observations in the 'Analysis' section. Then, move on to the judgement phase, drawing conclusions or making decisions based on your analysis. Format your findings in a JSON format with two keys: 'Analysis' for your insights on the responses and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "Vision Expert": """As a blind judge, you won't receive the figure from the user instructions, lacking direct visual context. An AI-generated analysis will be provided as optional supplementary information, but bear in mind its potential inaccuracies. Your primary task is to conduct a thorough analysis of the responses independently. Include your observations and interpretations in the 'Analysis' section. Following this, advance to the judgement phase, forming decisions based on your analysis, optionally informed by the AI analysis. Present your findings in a JSON format with keys 'Analysis' for your insights on the responses and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures."""}

tasks = {"score": """
You will receive a single response from the AI assistant to user's instruction. Use scores to show the quality of the response. Here is the detailed scoring rubric for evaluating the quality of responses from AI assistants:
Poor (1): The response significantly deviates from the user's instruction and fails to address the query effectively.
Fair (2): The response addresses the user's instruction partially, with evident shortcomings in relevance, accuracy, or comprehensiveness.
Average (3): The response adequately addresses the user's instruction, showing a fair level of relevance, accuracy, and comprehensiveness. 
Good (4): The response is well-aligned with the user's instruction, demonstrating a high degree of relevance, accuracy, and comprehensiveness.
Excellent (5): The response perfectly adheres to the user's instruction, excelling in relevance, accuracy, comprehensiveness, creativity, and granularity.
""",
"pair": """You will be presented with two responses from different assistants to the same user instruction.
Your task is to assess and compare these responses based on how effectively they adhere to the user's original instruction and how aptly they address the user's inquiry.
""",
"batch": """
Your task is to assess and compare these responses based on how effectively they adhere to the user's original instruction and how aptly they address the user's inquiry.
After your assessment and comparison, you should RANK the responses from best to worst as the following template. Rank four responses in the form "BACD", if B is the best, A is the second, C is the third, D is the worst."""}

object_judgement_prompt = """Do not allow the length of the responses to influence your evaluation.
Do not favor certain names or position of the assistants. Be as objective as possible."""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_json", type=str, default=None, help="The path to the JSON file containing the items to evaluate.")
    parser.add_argument("--output_json", type=str, default=None, help="The path to the JSONL file to write the results to.")
    parser.add_argument("--temperature", type=float, default=0.2, help="The temperature to use for inference.")
    parser.add_argument("--top_p", type=float, default=0.4, help="The top-p to use for inference.")
    parser.add_argument("--image_root", type=str, default=None, help="The root directory for the images.")
    parser.add_argument("--evaluate", type=str, default="score", help="The type of evaluation to perform")
    parser.add_argument("--setting", type=str, default="No COT", help="The setting of the evaluation")

    args = parser.parse_args()
    # Load the llava model
    model_path = "liuhaotian/llava-v1.5-13b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    # Load the JSON file
    items = []
    with open(args.input_json, "r") as json_file:
        for line in json_file:
            items.append(json.loads(line))

    # Open the JSONL file for writing
    with open(args.output_json, "a") as jsonl_file:
        # Iterate over each item with a progress bar
        for item in tqdm(items[:], desc="Processing items"):
            # Update the args with the current item's data
            prompt = start + "\nEvaluation Steps:\n" + settings[setting] + "\nEvaluation Method:\n" + tasks[evaluate] + "\nNotice:\n" + object_judgement_prompt + "\nHere is the input:\n"
            if args.evaluate == "score":  
                prompt += f"""
    Use "1", "2", "3", "4", "5" to indicate your evaluate score.
    [The Start of User Instruction]
    {item['instruction']}
    [The End of User Instruction]"""
                if args.setting == "Vision Expert":
                    prompt += f"""
    [The Start of Vision Expert’s Answer]
    {item['vision_information']}
    [The End of Vision Expert’s Answer]"""
                prompt += f"""
    [The Start of Assistant’s Answer]
    {item['answer']}
    [The End of Assistant’s Answer]"""
            elif args.evaluate == "pair":
                prompt += f"""
    Indicate your decision in the key 'Judgement', use "A" if assistant A prevails, "B" if assistant B does, and "C" for a tie.
    [The Start of User Instruction]
    {item['instruction']}
    [The End of User Instruction]"""
                if args.setting == "Vision Expert":
                    prompt += f"""
    [The Start of Vision Expert’s Answer]
    {item['vision_information']}
    [The End of Vision Expert’s Answer]"""
                prompt += f"""
    [The Start of Assistant A’s Answer]
    {item['answer1']['answer']}
    [The End of Assistant A’s Answer]
    [The Start of Assistant B’s Answer]
    {item['answer2']['answer']}
    [The End of Assistant B’s Answer]"""
            elif args.evaluate == "batch":
                prompt += "Indicate your final rank in the key 'Judgement'." + f"""
    [The Start of User Instruction]
    {item['instruction']}
    [The End of User Instruction]"""
                if args.setting == "Vision Expert":
                    prompt += f"""
    [The Start of Vision Expert’s Answer]
    {item['vision_information']}
    [The End of Vision Expert’s Answer]"""
                assistant_name = "A"
                for i in range(len(item['answers'])):
                    prompt += f"[The Start of Assistant {assistant_name}’s Answer]\n"
                    prompt += item['answers'][i]['answer'] + "\n"
                    prompt += f"[The End of Assistant {assistant_name}’s Answer]\n"
                    assistant_name = chr(ord(assistant_name) + 1)
            print(prompt)
            args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query":  prompt, 
                "conv_mode": None,
                "image_file": args.image_root + item['image_path'],  # Assuming each item has an 'image_file' field
                "sep": ",",
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_beams": 1,
                "max_new_tokens": 2048
            })()

            # Evaluate the model
            item['judge'] = eval_model(args, tokenizer, model, image_processor, context_len)
            
            # Write the item to the JSONL file
            jsonl_file.write(json.dumps(item) + "\n")
    