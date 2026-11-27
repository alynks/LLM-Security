
import torch
import json
from pathlib import Path
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Output directory
OUTPUT_DIR = Path("/data/sriparna/arnav/LLM Security/outputs/chat_logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create a timestamped log file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = OUTPUT_DIR / f"chat_{timestamp}.jsonl"

print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully!")
print("Device:", model.device)
print("Log file:", LOG_FILE)

while True:
    prompt = input("\nYou: ").strip()

    if prompt.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    if not prompt:
        continue

    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print("\nModel:", response)

    # Save prompt and response
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "prompt": prompt,
        "response": response
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[Saved to log]")

