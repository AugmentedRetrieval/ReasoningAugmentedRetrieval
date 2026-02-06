# Reasoning Augmented Retrieval

This is the official repository for the code used to generate data and train models in the paper Reasoning-Augmented Representations for Multimodal Retrieval.

---
## Training

Training code will be available soon.

---

## Data Generation

Please use files under `data` to generate enhanced queries and corpus entries. We use VLLM to expedite this process.

First, install the requirements:
```bash
cd data
pip install -r requirements.txt
```

Next, use `start_processes.py` to spawn VLLM processes to generate data. Our implementation spawns one worker on each GPU, and by default processes at most 60k items using one worker. After all VLLM processes are done, we merge the results in the end.
Example usage is as follows:
```bash
python start_processes.py [-h] [--num_gpus NUM_GPUS] [--num_threads_per_gpu NUM_THREADS_PER_GPU] [--max_num_items_per_gpu MAX_NUM_ITEMS_PER_GPU] [--model MODEL] --data_path DATA_PATH --image_root IMAGE_ROOT --output_path OUTPUT_PATH --type {query,cand_pool}
```

- `--num_gpus` defaults to 8.
- `--num_threads_per_gpu` depends on the total number of CPU cores on your machine. Default is 12 for a 128-core machine. Reduce if not enough cores.
- `--max_num_items_per_gpu` defaults to 60000. Works well on a machine with 1TB memory and 8 consecutive workers at a time. Reduce if RAM is fewer.
- `--model` defaults to `Qwen/Qwen3-VL-8B-Instruct`.
- `--data_path` is the path towards the specific file you want to process. Typically under `--image_root`.
- `--image_root` is the root folder of MBEIR images, or the entire MBEIR dataset you downloaded.
- `--output_path` specifies where you want to put the output.
- `--type` is either `query` or `cand_pool`, which is necessary for VLLM prompting.
