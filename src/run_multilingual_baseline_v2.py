
import json
import csv
import re
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = Path("/data/sriparna/arnav/LLM-Security")
MODEL_NAME = "Qwen/Qwen3Guard-Gen-0.6B"

DATASET = BASE / "datasets/multilingual_baseline_v2.jsonl"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
with open(DATASET, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

print("Dataset loaded:", len(records), "prompts")

# Load guard
print("\nLoading safety guard...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

print("Guard loaded on:", model.device)

results = []

# Evaluate prompts
for i, item in enumerate(records, 1):
    messages = [{"role": "user", "content": item["prompt"]}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    elapsed = time.perf_counter() - start

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    label_match = re.search(
        r"Safety:\s*(Safe|Unsafe|Controversial)",
        response,
        re.IGNORECASE
    )

    category_match = re.search(
        r"Categories:\s*(.*)",
        response,
        re.IGNORECASE
    )

    predicted = (
        label_match.group(1).capitalize()
        if label_match else "Unparsed"
    )

    categories = (
        category_match.group(1).strip()
        if category_match else "Unparsed"
    )

    result = {
        **item,
        "predicted_label": predicted,
        "predicted_categories": categories,
        "inference_time_seconds": round(elapsed, 4),
        "label_match": predicted == item["expected_label"],
        "raw_guard_response": response
    }

    results.append(result)

    print(
        f"[{i:03d}/{len(records)}] "
        f"{item['id']} | {item['language']} | "
        f"Expected: {item['expected_label']} | "
        f"Predicted: {predicted}"
    )

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = OUTPUT_DIR / f"baseline_v2_results_{timestamp}.csv"

with open(result_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Summary
print("\n" + "=" * 55)
print("DATASET V2 EVALUATION COMPLETED")
print("=" * 55)

correct = sum(r["label_match"] for r in results)
print(f"Overall agreement: {correct}/{len(results)} "
      f"({correct / len(results) * 100:.2f}%)")

for language in dict.fromkeys(r["language"] for r in results):
    subset = [r for r in results if r["language"] == language]
    matches = sum(r["label_match"] for r in subset)
    print(
        f"{language}: {matches}/{len(subset)} "
        f"({matches / len(subset) * 100:.2f}%)"
    )

print("\nResults saved to:", result_file)
