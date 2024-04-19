'''
Author: misaki misakiwang74@gmail.com
Date: 2024-04-19 11:41:03
LastEditors: misaki misakiwang74@gmail.com
LastEditTime: 2024-04-19 11:41:15
'''
import json
import os

def merge_and_sort_jsonl_files(directory_path, output_file):
    all_data = []

    # Iterate over all files in the given path
    for filename in os.listdir(directory_path):
        if filename.endswith('.jsonl'):
            # Read and parse each line for each jsonl file
            with open(os.path.join(directory_path, filename), 'r') as file:
                for line in file:
                    all_data.append(json.loads(line))
    
    # Sort all data by the 'pair_id' keyword
    sorted_data = sorted(all_data, key=lambda x: x['score_id'])
    print(len(all_data))
    # Write the sorted data into a new json file
    with open(output_file, 'w') as file:
        json.dump(sorted_data, file, ensure_ascii=False, indent=4)

# Usage example
merge_and_sort_jsonl_files('/media/ssd/cdp/MLLM-Judge/Dataset/Benchmark/arxiv_data/score/', 'output.json')
