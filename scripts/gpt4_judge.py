import base64
import requests
import json
from tqdm import tqdm  
from argparse import ArgumentParser
import time
import numpy as np

start = """Please serve as an unbiased judge in assessing the quality of the responses from AI assistants regarding the user's instruction and a figure. """

settings = {"COT figure": """Please examine the provided image attentively. Begin by conducting a comprehensive analysis of the figure provided. Detail your observations and insights in the 'Figure Analysis' section. Next, utilize the insights from your initial analysis to critically evaluate the responses. Summarize this evaluation in the 'Analysis' section. Finally, based on your figure analysis and response evaluation, form a well-reasoned judgement. Document this in the 'Judgement' section. Ensure that your final output in a JSON format with keys: 'Figure Analysis' for the initial figure assessment, 'Analysis' for the evaluation of responses against your ground truth, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "COT instruction": """Please examine the provided image attentively. Begin by providing a detailed response to the user instructions, treating this response as the baseline or 'ground truth'. This response will form the 'Response' section. Next, use this established ground truth to systematically analyze and evaluate the responses to the same instruction. This evaluation will form the 'Analysis' section. After the analysis, move forward to the judgement phase, where you will give final judgement based on the analysis of the responses compared to the ground truth. Give your judgement in the 'Judgement' section. Ensure that your final output is structured in a JSON format with the keys 'Response' for the answer to the instruction, 'Analysis' for the evaluation of responses, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "COT figure instruction": """Please examine the provided image attentively. Begin with an in-depth analysis of the figure. Detail your observations and insights in the 'Figure Analysis' section. Then, provide a detailed response to the user instructions, treating this as 'Response' and the ground truth. Next, compare and analyze the responses to the same instruction against your ground truth in the 'Analysis' section. Finally, give your final judgement in 'Judgement'. Structure your output in JSON format, with the following keys: 'Figure Analysis' for the initial figure assessment, 'Response' for your response to the instructions, 'Analysis' for the evaluation of responses against your ground truth, and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "No COT": """Please examine the provided image attentively. Begin by conducting a detailed analysis of the responses provided. Capture your comprehensive observations and insights in the 'Analysis' section. Following your analysis, move on to the judgement phase, where you will make informed decisions or conclusions based on the analysis conducted. Give your final judgements in the 'Judgement' section. Ensure that your final output with keys 'Analysis' for the initial response analysis, and 'Judgement' for your final judgement only.""",
            "No Figure": """As a blind judge, you will not have access to the figure mentioned in the user instructions. Your task is to impartially assess the responses based solely on the information presented within them, without visual context of the figure. Begin by performing a detailed analysis of the responses, capturing your observations in the 'Analysis' section. Then, move on to the judgement phase, drawing conclusions or making decisions based on your analysis. Format your findings in a JSON format with two keys: 'Analysis' for your insights on the responses and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures.""",
            "Vision Expert": """As a blind judge, you won't receive the figure from the user instructions, lacking direct visual context. An AI-generated analysis will be provided as optional supplementary information, but bear in mind its potential inaccuracies. Your primary task is to conduct a thorough analysis of the responses independently. Include your observations and interpretations in the 'Analysis' section. Following this, advance to the judgement phase, forming decisions based on your analysis, optionally informed by the AI analysis. Present your findings in a JSON format with keys 'Analysis' for your insights on the responses and 'Judgement' for your final judgement only. Ensure that the content under each key does not contain any nested JSON structures."""}

tasks = {"score": """
You will receive a single response from the AI assistant to user's instruction. Use scores to show the quality of the response. Here is the detailed scoring rubric for evaluating the quality of responses from AI assistants:
Poor (1): The response significantly deviates from the user's instruction and fails to address the query effectively. It shows a lack of relevance, accuracy, and comprehensiveness. Creativity and granularity are absent or poorly executed.
Fair (2): The response addresses the user's instruction partially, with evident shortcomings in relevance, accuracy, or comprehensiveness. It lacks depth in creativity and granularity, indicating a superficial understanding of the user's inquiry.
Average (3): The response adequately addresses the user's instruction, showing a fair level of relevance, accuracy, and comprehensiveness. It reflects a basic level of creativity and granularity but may lack sophistication or depth in fully capturing the user's inquiry.
Good (4): The response is well-aligned with the user's instruction, demonstrating a high degree of relevance, accuracy, and comprehensiveness. It shows creativity and a nuanced understanding of the topic, with detailed granularity that enhances the response quality.
Excellent (5): The response perfectly adheres to the user's instruction, excelling in relevance, accuracy, comprehensiveness, creativity, and granularity. It provides an insightful, detailed, and thorough answer, indicating a deep and nuanced understanding of the user's inquiry.
""",
"pair": """You will be presented with two responses from different assistants to the same user instruction.
Your task is to assess and compare these responses based on how effectively they adhere to the user's original instruction and how aptly they address the user's inquiry.
""",
"exchange_pair": """You will be presented with two responses from different assistants to the same user instruction.
Your task is to assess and compare these responses based on how effectively they adhere to the user's original instruction and how aptly they address the user's inquiry.
""",
"batch": """You will be presented with several responses from different assistants to the same user instruction.
Your task is to assess and  compare these responses based on how effectively they adhere to the user's original instruction and how aptly they address the user's inquiry.
After your assessment and comparison, you should RANK the responses from best to worst as the following template. If Assistant A is the best response, Assistant D is the worst response, you should output like [[A]], [[B]], [[C]], [[D]]""",
"shuffled_batch": """You will be presented with several responses from different assistants to the same user instruction.
Your task is to assess and  compare these responses based on how effectively they adhere to the user's original instruction and how aptly they address the user's inquiry.
After your assessment and comparison, you should RANK the responses from best to worst as the following template. If Assistant A is the best response, Assistant D is the worst response, you should output like [[A]], [[B]], [[C]], [[D]]"""}

object_judgement_prompt = """Your assessment should identify whether the assistant effectively adheres to the user's instruction and addresses the user's inquiry.
In your evaluation, weigh factors such as relevance, accuracy, comprehensiveness, creativity, and the granularity of the responses.
Do not allow the length of the responses to influence your evaluation.
Do not favor certain names or position of the assistants. Be as objective as possible."""

if __name__ == "__main__":
    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument("--api", type=str, default=None, help="The API key to use for the OpenAI API")
    parser.add_argument("--json_file_path", type=str, default=None, help="The path to the JSON file containing the items to evaluate")
    parser.add_argument("--output_file_path", type=str, default=None, help="The path to the JSONL file to write the results to")
    parser.add_argument("--evaluate", type=str, default="score", help="The type of evaluation to perform")
    parser.add_argument("--setting", type=str, default="No COT", help="The setting of the evaluation")
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview", help="The model to use for the OpenAI API")
    parser.add_argument("--temperature", type=float, default=0.9, help="The temperature to use for the OpenAI API")
    parser.add_argument("--top_p", type=float, default=0.9, help="The top_p to use for the OpenAI API")
    args = parser.parse_args()
    api_key = args.api

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Load the JSON file
    if args.json_file_path[-1] == "l":
        items = []
        with open(args.json_file_path, "r") as jsonl_file:
            for line in jsonl_file:
                items.append(json.loads(line))
    else:
        with open(args.json_file_path, "r") as json_file:
            items = json.load(json_file)

    # Headers for the request
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Open the JSONL file for writing
    with open(args.output_file_path, "a") as jsonl_file:
        # Iterate over each item with a progress bar
        for item in tqdm(items[args.start:args.end], desc="Processing items"):
            image_path = "/media/ssd/cdp/LMM-as-a-Judge/lmm_judge_dataset/" + item['image_path']
            base64_image = encode_image(image_path)
            prompt = start + "\nEvaluation Steps:\n" + settings[args.setting] + "\nEvaluation Method:\n" + tasks[args.evaluate] + "\nNotice:\n" + object_judgement_prompt + "\nHere is the input:\n"
            if args.evaluate == "score":  
                prompt += f"""
Use "[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]" to indicate your evaluate score in the key 'Judgement'.
[The Start of User Instruction]
{item['instruction']}
[The End of User Instruction]"""
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
                prompt += "Indicate your final rank in the key 'Judgement'." + f"""
[The Start of User Instruction]
{item['instruction']}
[The End of User Instruction]"""
                assistant_name = "A"
                num_assistant = 0
                for i in range(3):
                    prompt += f"[The Start of Assistant {assistant_name}’s Answer]\n"
                    prompt += item[f'answer{num_assistant}']['answer'] + "\n"
                    prompt += f"[The End of Assistant {assistant_name}’s Answer]\n"
                    assistant_name = chr(ord(assistant_name) + 1)
                    num_assistant += 1
                if 'answer3' in item.keys():
                    prompt += f"[The Start of Assistant {assistant_name}’s Answer]\n"
                    prompt += item['answer3']['answer'] + "\n"
                    prompt += f"[The End of Assistant {assistant_name}’s Answer]\n"
            elif args.evaluate == "shuffled_batch":
                prompt += "Indicate your final rank in the key 'Judgement'." + f"""
[The Start of User Instruction]
{item['instruction']}
[The End of User Instruction]"""
                if args.setting == "Vision Expert":
                    prompt += f"""
[The Start of Vision Expert’s Answer]
{item['vision_information']}
[The End of Vision Expert’s Answer]"""
                shuffled_index = np.arange(len(item['answers']))
                np.random.shuffle(shuffled_index)
                assistant_name = "A"
                for i in range(len(shuffled_index)):
                    prompt += f"[The Start of Assistant {assistant_name}’s Answer]\n"
                    prompt += item['answers'][shuffled_index[i]]['answer'] + "\n"
                    prompt += f"[The End of Assistant {assistant_name}’s Answer]\n"
                    assistant_name = chr(ord(assistant_name) + 1)
                item['shuffled_index'] = shuffled_index.tolist()
            payload = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": args.temperature,
                "top_p": args.top_p
            }
            success = False
            attempt = 0
            while not success and attempt < 3:
                response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
                # Check if there's an 'error' key in the response
                if 'error' in response.json():
                    print(f"Error encountered: {response.json()['error']}. Retrying in 20 seconds...")
                    attempt += 1
                    time.sleep(20)
                else:
                    success = True
            
            if success:
                # Parse the response and add it to the item
                item['new_judge'] = response.json()
                print(response.json())
                # Write the item to the JSONL file
                jsonl_file.write(json.dumps(item) + "\n")
            else:
                print(f"Failed to process item after 3 attempts: {item['id']}")
                # Optionally add logic to handle the failure case,
