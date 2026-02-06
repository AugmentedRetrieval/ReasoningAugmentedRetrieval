import argparse
import json
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=8)
parser.add_argument('--num_threads_per_gpu', type=int, default=12)
parser.add_argument('--max_num_items_per_gpu', type=int, default=60000)
parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--image_root', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--type', type=str, choices=["query", "cand_pool"], required=True)

args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

# 1. Create a Queue to manage GPU availability
# This queue acts as a token bucket. If it's empty, threads will wait here.
gpu_queue = queue.Queue()
for i in range(args.num_gpus):
    gpu_queue.put(i)

def run_infer(start, end):
    """
    Worker function that acquires a GPU, runs the task, and releases the GPU.
    """
    # Block until a GPU is available
    gpu_id = gpu_queue.get() 

    try:
        print(f"Assigning [{start}:{end}] to GPU {gpu_id}")

        cmd = (
            f"OMP_NUM_THREADS={args.num_threads_per_gpu} "
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            f"python infer.py {start} {end} --model {args.model} --data_path {args.data_path} --image_root {args.image_root} --output_path {args.output_path} --type {args.type}"
        )

        # Run the command and wait for it to finish
        subprocess.call(cmd, shell=True)
        
    except Exception as e:
        print(f"Error processing {e}")

    finally:
        # CRITICAL: Always return the GPU to the queue, even if the job fails
        gpu_queue.put(gpu_id)
        gpu_queue.task_done()


# 2. Use ThreadPoolExecutor
# We set max_workers to NUM_GPUS (or slightly higher) so we have threads ready 
# to grab a GPU as soon as one becomes free.
with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:

    # 3. Analyze file to see if it needs processing and how many chunks
    # We read line by line to avoid loading massive files entirely into RAM
    all_items = []

    with open(args.data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            all_items.append(item)

    total_items = len(all_items)
    print(f"Processing {total_items} items")

    # 4. Submit chunks to the executor
    # The executor handles the scheduling; the worker function handles the GPU assignment.
    for start in range(0, total_items, args.max_num_items_per_gpu):
        end = min(start + args.max_num_items_per_gpu, total_items)
        executor.submit(run_infer, start, end)

# after VLLM finishes inference, merge the results
result = []
target_filename = args.data_path.split('/')[-1].replace('.jsonl', '')

for filename in tqdm(os.listdir(args.output_path)):
    if target_filename not in filename or "_new.jsonl" in filename: continue

    with open(os.path.join(args.output_path, filename), 'r') as f:
        for line in f.readlines():
            res = json.loads(line)

            if "rewritten_query" in res:
                res["query_txt" if args.type == "query" else "txt"] = res["rewritten_query"]
                del res["rewritten_query"]
            
            result.append(res)
    # delete the intermediate file
    os.remove(os.path.join(args.output_path, filename))

with open(f"{args.output_path}/{target_filename}_new.jsonl", 'w') as f:
    for res in result:
        f.write(json.dumps(res) + "\n")
