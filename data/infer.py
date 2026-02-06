import argparse
import json
import os
import sys

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
parser.add_argument('--data_path', type=str)
parser.add_argument('--image_root', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--type', type=str, choices=["query", "cand_pool"])

args = parser.parse_args()


def infer(inputs, sampling_params, valid_indices, all_items, llm):
    # --- 4. Run Inference (Batch Processing) ---
    print(f"Running inference on {len(inputs)} items...")
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # --- 5. Process Results ---
    results = []
    processed_orig_indices = set()

    for i, output in enumerate(outputs):
        original_idx = valid_indices[i]
        generated_text = output.outputs[0].text
        
        # Store result back into a structured format
        result_item = all_items[original_idx].copy()
        result_item['rewritten_query'] = generated_text
        results.append(result_item)
        processed_orig_indices.add(original_idx)
    for i in range(len(all_items)):
        if i not in processed_orig_indices:
            results.append(all_items[i])

    # Optional: Save to file
    target_filename = args.data_path.split('/')[-1].replace('.jsonl', '')
    with open(f"{args.output_path}/{target_filename}_{sys.argv[1]}_{sys.argv[2]}.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

# --- 1. Load the Data ---
print(f"Loading data from {args.data_path}...")
all_items = []

with open(args.data_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        all_items.append(item)

print(f"Total items loaded: {len(all_items)}")


all_items = all_items[int(sys.argv[1]):int(sys.argv[2])]
print(f"Processing items from index {sys.argv[1]} to {sys.argv[2]}")

# --- 2. Initialize vLLM ---
llm = LLM(
    model=args.model, 
    trust_remote_code=True,
    max_model_len=16384, 
    limit_mm_per_prompt={"image": 1},
)

sampling_params = SamplingParams(max_tokens=128 if args.type == "query" else 512)

# --- 3. Prepare Inputs for vLLM ---
inputs = []
valid_indices = []

print("Preparing inputs and loading images...")
num_processed = 0

for idx, item in enumerate(tqdm(all_items)):
    # Extract Data
    query_txt = item.get('query_txt' if args.type == "query" else "txt", "")
    rel_img_path = item.get('query_img_path' if args.type == "query" else "img_path", "")
    
    # Handle missing text (default to empty string if None)
    if query_txt is None:
        query_txt = ""

    # Skip if no image path is provided
    if not rel_img_path:
        continue

    # construct absolute image path
    full_img_path = os.path.join(args.image_root, rel_img_path)

    # Check if image file actually exists
    if not os.path.exists(full_img_path):
        print(f"Warning: Image not found at {full_img_path}")
        continue

    try:
        # Load image using PIL
        image = Image.open(full_img_path).convert("RGB")
        use_image = True
        if query_txt != "" and args.type == "query":
            if query_txt.endswith('?') or query_txt.lower().startswith("what") or query_txt.lower().startswith("where") or query_txt.lower().startswith("why") or query_txt.lower().startswith("when") or query_txt.lower().startswith("which") or query_txt.lower().startswith("how"):
                # ---------------------------------------------------------
                # SCENARIO A: QUESTION / ENTITY RETRIEVAL
                # Strategy: Strict Entity Naming or Minimal Visual Anchors
                # ---------------------------------------------------------
                prompt = (
                    "Task: Rewrite the user's question by integrating the visual subject. \n"
                    "Goal: Create a search query that matches text documents. Keep it extremely concise.\n"
                    "\n"
                    "STRICT CONSTRAINT RULES:\n"
                    "1. **Length Limit:** The added visual description must be MAX 3-5 words. No long sentences.\n"
                    "2. **The 'Specific vs. Generic' Split:**\n"
                    "   - **If Unique Entity (Landmark, Art, Car Model):** Use the NAME only. Delete all visual adjectives.\n"
                    "     * BAD: 'Who built this tall iron tower?'\n"
                    "     * GOOD: 'Who built the Eiffel Tower?'\n"
                    "   - **If Generic Object (Food, Plant, Animal):** Use [Dominant Color/Material] + [Broad Category].\n"
                    "     * BAD: 'What is this delicious spicy red soup with shrimp?' (Too many distractors)\n"
                    "     * GOOD: 'What is this red noodle soup with shrimp?' (Anchors only)\n"
                    "3. **No 'Filler' Adjectives:** Banned words: 'beautiful', 'large', 'small', 'generic', 'distinct', 'looking', 'shaped'.\n"
                    "4. **No Environment:** Never mention background, weather, or lighting.\n"
                    "5. **Zero-Leakage:** NEVER answer the question yourself. YOU ARE ONLY REWRITING THE QUERY.\n"
                    "\n"
                    "EXAMPLES:\n"
                    "Input: [Photo of Giant Panda] | Query: 'When was it discovered?'\n"
                    "Output: When was the Giant Panda discovered?\n"
                    "(Reason: Named entity. No adjectives needed.)\n"
                    "\n"
                    "Input: [Photo of Yellowjacket Wasp] | Query: 'What species is this?'\n"
                    "Output: What species is this black and yellow wasp?\n"
                    "(Reason: 'Black and yellow' distinguishes it. 'Insect' is too broad, 'Wasp' is better.)\n"
                    "\n"
                    "Input: [Photo of Red Laksa Soup] | Query: 'What dish is this?'\n"
                    "Output: What dish is this red noodle soup with shrimp?\n"
                    "(Reason: 'Red', 'Noodle', 'Shrimp' are the only keys needed to find the recipe.)\n"
                    "\n"
                    "Input: [Photo of Blue Ford Focus] | Query: 'What car is this?'\n"
                    "Output: What car is this blue hatchback?\n"
                    "(Reason: 'Blue' and 'Hatchback' filter the candidates. 'Ford Focus' might be a hallucination, so we play it safe. We also don't want to leak the answer.)\n"
                    "\n"
                    "Input: [Photo of Melting Clock Painting] | Query: 'Who painted this?'\n"
                    "Output: Who painted The Persistence of Memory?\n"
                    "(Reason: Unique Art -> Specific Name.)\n"
                    "\n"
                    f"Current Task:\nQuery: {query_txt}\nInput Image:"
                )

            else:
                use_image = False
                # ---------------------------------------------------------
                # SCENARIO B: MODIFICATION / RELATIVE RETRIEVAL
                # Strategy: Syntactic Denoising (Garbage In -> Gold Out)
                # ---------------------------------------------------------
                prompt = (
                    "Task: Extract the key semantic phrases describing the TARGET image. Remove conversational filler and grammar words.\n"
                    "Input: User Query (describing a change or a target attribute).\n"
                    "\n"
                    "STRICT REDUCTION RULES:\n"
                    "1. **Delete Filler Verbs:** Remove 'Is', 'Has', 'Make', 'Change', 'Show', 'Put', 'Be'.\n"
                    "2. **Delete Pronouns/Articles:** Remove 'it', 'the', 'a', 'an', 'my', 'me', 'them', 'this'.\n"
                    "3. **Preserve Adjectives & Nouns:** Keep ALL descriptors (colors, patterns, objects). If the user says 'Is white', output 'White'.\n"
                    "4. **Preserve Prepositions:** Keep 'with', 'on', 'in', 'without' to maintain spatial/compositional logic.\n"
                    "There are cases where the original query is concise enough, and you might not have to change anything.\n"
                    "EXAMPLES:\n"
                    "Input: 'Is shiny and silver with shorter sleeves.'\n"
                    "Output: Shiny silver with shorter sleeves\n"
                    "\n"
                    "Input: 'Is white in color with short sleeves and is more plain.'\n"
                    "Output: White, short sleeves, more plain\n"
                    "\n"
                    "Input: 'Remove the lemon.'\n"
                    "Output: Remove lemon\n"
                    "\n"
                    "Input: 'Make the needle upside down in the hand.'\n"
                    "Output: Needle upside down in hand\n"
                    "\n"
                    "Input: 'Human and one animal from a different species.'\n"
                    "Output: Human and animal from different species\n"
                    "\n"
                    "Input: 'Is a plain white feminine t shirt and is a tan shirt.'\n"
                    "Output: Plain white feminine t-shirt and tan shirt\n"
                    "\n"
                    "Input: 'Remove all cheetahs.'\n"
                    "Output: Remove all cheetahs\n"
                    "\n"
                    "Input: 'Remove one cheetah.'\n"
                    "Output: Remove one cheetah.\n"
                    "\n"
                    "Input: 'Remove green from the background.'\n"
                    "Output: Remove green from background.\n"
                    "\n"
                    f"Current Input: {query_txt}"
                )
        else:
            num_words = 50 if args.type == "query" else 100
            prompt = (
                "Task: Generate a precise, keyword-rich search query based on the [Image].\n"
                "\n"
                "Instructions:\n"
                "1. **Subject First:** Identify the main object, entity, or scene layout immediately.\n"
                "2. **Distinctive Features:** List specific details: colors, materials, text/logos (if visible), and unique shapes. If a detail doesn't exist, don't mention it (no stating 'no visible logos or text').\n"
                "3. **Entity Recognition:** If the object is a named entity (e.g., 'Eiffel Tower', 'Toyota Camry', 'Nike'), state it.\n"
                "4. **Viewpoint:** Mention the angle (e.g., 'close-up', 'aerial', 'profile') ONLY IF it distinguishes the image.\n"
                "5. **No Filler:** Do not use aesthetic words (e.g., 'beautiful', 'cinematic'). Focus on factual visual content.\n"
                f"6. **Length:** Maximum {num_words} words.\n"
                "Reference Image:"
            )
        
        # --- Construct Prompt ---
        
        if use_image:
            prompt_text = (
                "<|im_start|>system\nYou are an expert in refining queries for multimodal retrieval.<|im_end|>\n" if args.type == "query" else "<|im_start|>system\nYou are an expert in writing database text entries for images for multimodal retrieval.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{prompt}<|vision_start|><|image_pad|><|vision_end|>\nOutput: <|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            # Prepare vLLM input dictionary
            inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {
                    "image": image
                },
            })
        else:
            prompt_text = (
                "<|im_start|>system\nYou are an expert in refining queries for multimodal retrieval.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            # Prepare vLLM input dictionary
            inputs.append({
                "prompt": prompt_text,
            })
        valid_indices.append(idx)
        
    except Exception as e:
        print(f"Error loading image {full_img_path}: {e}")
        continue


if __name__ == "__main__":
    infer(inputs, sampling_params, valid_indices, all_items, llm)