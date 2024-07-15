import os
import random

# Define file paths
input_dir = '../Dataset/Benchmark'
output_dir = '../Dataset/Benchmark_Lite'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_batch_file(input_file, output_file):
    """ Directly save batch.jsonl file """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(line)

def process_pair_file(input_file, output_file):
    """ Randomly save one out of every 6 in groups while preserving original order """
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    original_count = len(lines)
    sampled_lines = []

    for i in range(0, original_count, 6):
        group = lines[i:i + 6]
        if group:
            sampled_lines.append(random.choice(group))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in sampled_lines:
            outfile.write(line)
    return original_count, len(sampled_lines)

def process_score_file(input_file, output_file):
    """ Randomly save one out of every 4 in groups while preserving original order """
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    original_count = len(lines)
    sampled_lines = []

    for i in range(0, original_count, 4):
        group = lines[i:i + 4]
        if group:
            sampled_lines.append(random.choice(group))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in sampled_lines:
            outfile.write(line)
    return original_count, len(sampled_lines)

def main():
    # Define input and output file paths
    batch_file = os.path.join(input_dir, 'batch.jsonl')
    pair_file = os.path.join(input_dir, 'pair.jsonl')
    score_file = os.path.join(input_dir, 'score.jsonl')
    
    batch_output_file = os.path.join(output_dir, 'batch_lite.jsonl')
    pair_output_file = os.path.join(output_dir, 'pair_lite.jsonl')
    score_output_file = os.path.join(output_dir, 'score_lite.jsonl')
    
    # Process files and get counts
    process_batch_file(batch_file, batch_output_file)
    pair_counts = process_pair_file(pair_file, pair_output_file)
    score_counts = process_score_file(score_file, score_output_file)
    
    print(f"Pair file: {pair_counts[0]} JSON objects in source, {pair_counts[1]} JSON objects in processed file")
    print(f"Score file: {score_counts[0]} JSON objects in source, {score_counts[1]} JSON objects in processed file")
    print("Files processed and saved to", output_dir)

if __name__ == '__main__':
    main()
