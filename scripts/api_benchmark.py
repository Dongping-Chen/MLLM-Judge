import torch
from argparse import ArgumentParser
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import json
from tqdm import tqdm
import os
from prompt import get_prompt
from get_vlm_res import gpt_4v, gpt_4o, llava_1_6_34b, llava_1_6_13b, llava_1_6_7b, qwen_vl_plus, qwen_vl_max

import time

def retry(attempts=3, delay=10):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i+1}/{attempts} failed: {e}")
                    if i < attempts - 1:
                        time.sleep(delay)  # Wait a bit before retrying
            return None
        return wrapper
    return decorator


@retry(3)
def get_res(model, image_path, prompt, api, temperature, top_p, llm_assist: bool=False):
    func_name = model.replace("-", "_").replace(".", "_")
    func = globals().get(func_name)
    output = func(image_path, prompt, api, temperature, top_p)
    try:
        json_output = json.loads(output)
        judgement = json_output["Judgement"]
        return judgement
    except KeyError:
        if llm_assist:
            pass
        return output
    


def construct_input(prompt_dict, judge_mode, setting, instruction, responses):
    prompt = prompt_dict["start"] + "\nEvaluation Steps:\n" + prompt_dict["settings"][setting] + "\nEvaluation Method:\n" + prompt_dict["tasks"][judge_mode] + "\nNotice:\n" + prompt_dict["notice"] + "\nHere is the input:\n"
    if judge_mode == "score":
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]
[The Start of Assistant’s Answer]
{responses[0]}
[The End of Assistant’s Answer]""" 
    elif judge_mode == 'pair':
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]
[The Start of Assistant A’s Answer]
{responses[0]}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{responses[1]}
[The End of Assistant B’s Answer]"""
    elif judge_mode == 'batch':
        prompt += f"""
[The Start of User Instruction]
{instruction}
[The End of User Instruction]"""
        assistant_name = "A"
        num_assistant = 0
        for i in range(len(responses)):
            prompt += f"[The Start of Assistant {assistant_name}’s Answer]\n"
            prompt += responses[num_assistant] + "\n"
            prompt += f"[The End of Assistant {assistant_name}’s Answer]\n"
            assistant_name = chr(ord(assistant_name) + 1)
            num_assistant += 1
    return prompt
    
def benchmark(model, judge_mode, setting, api, image_root, temperature, top_p):
    items = []
    with open(f"../Dataset/Benchmark/{judge_mode}.jsonl", "r") as json_file:
        for line in json_file:
                items.append(json.loads(line))
    output_path = f"./benchmark_result/{judge_mode}_{model}.jsonl"
    prompt_dict = get_prompt()
    for item in tqdm(items[:], desc="Processing items"):
        image_path = image_root + item['image_path']  
        prompt = construct_input(prompt_dict, judge_mode, setting, item['instruction'])

        response = get_res(model, image_path, prompt, api, temperature, top_p)
        item['mllm_judge'] = response
        with open(output_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(item) + "\n")
        

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="The path to the JSON file containing the items to evaluate.")
    parser.add_argument("--judge_mode", type=str, default=None, help="The path to the JSON file containing the items to evaluate.")
    parser.add_argument("--temperature", type=float, default=0.2, help="The temperature to use for inference.")
    parser.add_argument("--top_p", type=float, default=0.4, help="The top-p to use for inference.")
    parser.add_argument("--image_root", type=str, default=None, help="The root directory for the images.")
    parser.add_argument("--setting", type=str, default="No COT", help="The setting of the evaluation")
    parser.add_argument("--api", type=str, default=None, help="API for inference.")
    args = parser.parse_args()
    assert args.judge_mode not in ['score', 'batch', 'pair'], "Invalid judge mode"
    assert args.model not in ['gemini', 'gpt-4v', 'gpt-4o', 'llava-1.6-34b', 'llava-1.6-13b', 'llava-1.6-7b', 'qwen-vl-plus', 'qwen-vl-max', 'qwen-vl-chat']
    
    benchmark = benchmark(args.model, args.judge_mode, args.setting, args.api, args.model_ckpt, args.image_root, args.temperature, args.top_p)
    

if __name__ == '__main__':
    main()