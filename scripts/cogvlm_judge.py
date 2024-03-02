import torch
from argparse import ArgumentParser
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import json
from tqdm import tqdm

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
After your assessment and comparison, you should RANK the responses from best to worst as the following template. Rank four responses in the form "ABCD", if A is the best, B is the second, C is the third, D is the worst."""}

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
    parser.add_argument("--model_ckpt", type=str, default=None, help="The path to the model checkpoint to use.")
    args = parser.parse_args()
    
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    device_map = infer_auto_device_map(model, max_memory={0:'18GiB',1:'18GiB','cpu':'40GiB'}, no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
    model = load_checkpoint_and_dispatch(
        model,
        args.model_ckpt,
        device_map=device_map,
    )
    model = model.eval()

    # check device for weights if u want to
    for n, p in model.named_parameters():
        print(f"{n}: {p.device}")

    # Load the JSON file
    items = []
    with open(args.input_json, "r") as json_file:
        for line in json_file:
                items.append(json.loads(line))

    with open("CogVLM_score_append.jsonl",'a') as jsonl_file:
        for item in tqdm(items[:], desc="Processing items"):
            image_path = args.image_root + item['image_path']
            # chat example
            prompt = start + "\nEvaluation Steps:\n" + settings[args.setting] + "\nEvaluation Method:\n" + tasks[args.evaluate] + "\nNotice:\n" + object_judgement_prompt + "\nHere is the input:\n"
            if args.evaluate == "score":  
                prompt += f"""
    Use "1", "2", "3", "4", "5" to indicate your evaluate score directly.
    [The Start of User Instruction]
    {item['instruction']}
    [The End of User Instruction]
    Use "1", "2", "3", "4", "5" to indicate your evaluate score directly."""
                prompt += f"""
    [The Start of Assistant’s Answer]
    {item['answer']}
    [The End of Assistant’s Answer]"""
            elif args.evaluate == "pair":
                prompt += f"""
    Indicate your decision in the key 'Judgement', use "[[A]]" if assistant A prevails, "[[B]]" if assistant B does, and "[[C]]" for a tie.
    [The Start of User Instruction]
    {item['instruction']}
    [The End of User Instruction]"""
                prompt += f"""
    [The Start of Assistant A’s Answer]
    {item['answer1']['answer']}
    [The End of Assistant A’s Answer]
    [The Start of Assistant B’s Answer]
    {item['answer2']['answer']}
    [The End of Assistant B’s Answer]"""
            elif args.evaluate == "batch":
                prompt += """Indicate your final rank in the key 'Judgement'. Rank four responses in the form "BACD", if B is the best, A is the second, C is the third, D is the worst.""" + f"""
    [The Start of User Instruction]
    {item['instruction']}
    [The End of User Instruction]"""
                assistant_name = "A"
                for i in range(len(item['answers'])):
                    prompt += f"[The Start of Assistant {assistant_name}’s Answer]\n"
                    prompt += item['answers'][i]['answer'] + "\n"
                    prompt += f"[The End of Assistant {assistant_name}’s Answer]\n"
                    assistant_name = chr(ord(assistant_name) + 1)
            image = Image.open(image_path).convert('RGB')
            inputs = model.build_conversation_input_ids(tokenizer, query=prompt, history=[], images=[image])  # chat mode
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
                'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
            }
            gen_kwargs = {"max_length": 2048, "do_sample": True, "temperature": args.temperature, "top_p": args.top_p}

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
            print(outputs.shape)
            response = tokenizer.decode(outputs[0] if len(outputs) >=1 else outputs)
            print(response)
            # Parse the response and add it to the item
            item['gpt_output'] = response
            
            # Write the item to the JSONL file
            jsonl_file.write(json.dumps(item) + "\n")
