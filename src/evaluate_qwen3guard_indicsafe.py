#!/usr/bin/env python3
"""Run Qwen3Guard-Gen on an IndicSafe JSONL prompt-evaluation split.

Prompt-only moderation: each record is sent as one user message. No assistant
response or extra evaluation instruction is added to the prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_DEFAULT = "Qwen/Qwen3Guard-Gen-0.6B"
CATEGORY_NAMES = [
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
    "None",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3Guard-Gen on an IndicSafe prompt-only JSONL split."
    )
    parser.add_argument("--input", required=True, help="Path to evaluation_records.jsonl")
    parser.add_argument(
        "--output-dir",
        default="results/qwen3guard_indicsafe_baseline",
        help="Directory for predictions and summaries",
    )
    parser.add_argument("--model", default=MODEL_DEFAULT, help="Hugging Face model ID or local path")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N smoke test")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            if not row.get("record_id") or not row.get("prompt"):
                raise ValueError(
                    f"Line {line_number} must contain non-empty 'record_id' and 'prompt'."
                )
            rows.append(row)
    return rows


def parse_guard_output(raw_output: str) -> tuple[str | None, list[str], str | None]:
    safety_match = re.search(
        r"Safety\s*:\s*(Safe|Unsafe|Controversial)\b",
        raw_output,
        flags=re.IGNORECASE,
    )
    label = safety_match.group(1).capitalize() if safety_match else None

    categories_match = re.search(
        r"Categories\s*:\s*([^\r\n]*)",
        raw_output,
        flags=re.IGNORECASE,
    )
    categories_text = categories_match.group(1).strip() if categories_match else None
    categories: list[str] = []
    if categories_text:
        for item in categories_text.split(","):
            candidate = item.strip()
            for known in CATEGORY_NAMES:
                if candidate.casefold() == known.casefold() and known not in categories:
                    categories.append(known)

    return label, categories, categories_text


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_summaries(predictions_path: Path, output_dir: Path) -> None:
    # If a run was resumed, keep the latest record for each record_id.
    latest: dict[str, dict[str, Any]] = {}
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            latest[item["record_id"]] = item

    completed = [item for item in latest.values() if item.get("status") == "ok"]
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)

    for item in completed:
        label = item.get("safety_label") or "UNPARSED"
        by_language[item.get("language") or "UNKNOWN"][label] += 1
        by_category[item.get("canonical_category") or "CATEGORY_UNAVAILABLE"][label] += 1

    labels = ["Safe", "Unsafe", "Controversial", "UNPARSED"]
    language_rows = []
    for language in sorted(by_language):
        counts = by_language[language]
        total = sum(counts.values())
        row: dict[str, Any] = {"language": language, "total": total}
        for label in labels:
            row[label.lower()] = counts.get(label, 0)
            row[label.lower() + "_percent"] = round(100 * counts.get(label, 0) / total, 2) if total else 0.0
        language_rows.append(row)

    category_rows = []
    for category in sorted(by_category):
        counts = by_category[category]
        total = sum(counts.values())
        row = {"canonical_category": category, "total": total}
        for label in labels:
            row[label.lower()] = counts.get(label, 0)
            row[label.lower() + "_percent"] = round(100 * counts.get(label, 0) / total, 2) if total else 0.0
        category_rows.append(row)

    write_csv(
        output_dir / "summary_by_language.csv",
        language_rows,
        ["language", "total"] + [field for label in labels for field in (label.lower(), label.lower() + "_percent")],
    )
    write_csv(
        output_dir / "summary_by_category.csv",
        category_rows,
        ["canonical_category", "total"] + [field for label in labels for field in (label.lower(), label.lower() + "_percent")],
    )

    summary = {
        "unique_records_in_output": len(latest),
        "completed_records": len(completed),
        "records_with_parsed_safety_label": sum(1 for x in completed if x.get("safety_label")),
        "records_without_parsed_safety_label": sum(1 for x in completed if not x.get("safety_label")),
        "safety_label_counts": dict(Counter(x.get("safety_label") or "UNPARSED" for x in completed)),
        "note": (
            "Descriptive guard-output distributions only. IndicSafe source categories are not "
            "treated as ground-truth Safe/Unsafe labels; no accuracy, FPR, FNR, or ASR is computed."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.is_file():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit <= 0:
        print("ERROR: --limit must be a positive integer.", file=sys.stderr)
        return 2
    if args.max_new_tokens <= 0:
        print("ERROR: --max-new-tokens must be positive.", file=sys.stderr)
        return 2

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available in this process. Start the H100 MIG allocation "
            "and activate the correct environment first.",
            file=sys.stderr,
        )
        return 2

    records = load_jsonl(input_path)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print("ERROR: No evaluation records found.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    completed_ids: set[str] = set()
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("status") == "ok":
                    completed_ids.add(item["record_id"])

    print(f"Input records selected : {len(records)}")
    print(f"Already completed      : {len(completed_ids)}")
    print(f"Pending records        : {sum(1 for r in records if r['record_id'] not in completed_ids)}")
    print(f"Model                  : {args.model}")
    print(f"GPU                    : {torch.cuda.get_device_name(0)}")
    print("Loading tokenizer and model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    run_metadata = {
        "model": args.model,
        "input_file": str(input_path),
        "selected_input_records": len(records),
        "max_new_tokens": args.max_new_tokens,
        "method": "prompt-only; one user message per record; deterministic generation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    attempted = 0
    with predictions_path.open("a", encoding="utf-8") as output_handle:
        for index, row in enumerate(records, start=1):
            record_id = str(row["record_id"])
            if record_id in completed_ids:
                continue

            messages = [{"role": "user", "content": str(row["prompt"])}]
            started = time.perf_counter()
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                )
                model_inputs = tokenizer(
                    [formatted_prompt],
                    return_tensors="pt",
                ).to(model.device)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                new_tokens = generated_ids[0][model_inputs.input_ids.shape[-1] :]
                raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                label, categories, category_text = parse_guard_output(raw_output)

                result = {
                    "record_id": record_id,
                    "seed_group_id": row.get("seed_group_id"),
                    "record_type": row.get("record_type"),
                    "prompt": row.get("prompt"),
                    "prompt_en": row.get("prompt_en"),
                    "language": row.get("language"),
                    "language_code": row.get("language_code"),
                    "script": row.get("script"),
                    "canonical_category": row.get("canonical_category"),
                    "raw_category": row.get("raw_category"),
                    "split": row.get("split"),
                    "model": args.model,
                    "safety_label": label,
                    "categories": categories,
                    "categories_raw": category_text,
                    "raw_output": raw_output,
                    "latency_seconds": round(time.perf_counter() - started, 4),
                    "status": "ok",
                    "error": None,
                }
            except Exception as exc:
                result = {
                    "record_id": record_id,
                    "seed_group_id": row.get("seed_group_id"),
                    "language": row.get("language"),
                    "canonical_category": row.get("canonical_category"),
                    "model": args.model,
                    "safety_label": None,
                    "categories": [],
                    "raw_output": None,
                    "latency_seconds": round(time.perf_counter() - started, 4),
                    "status": "error",
                    "error": repr(exc),
                }

            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
            attempted += 1

            if attempted == 1 or attempted % 5 == 0:
                print(
                    f"Processed {attempted} new records "
                    f"(input position {index}/{len(records)}): "
                    f"{result.get('language')} -> {result.get('safety_label') or result.get('status')}",
                    flush=True,
                )

    create_summaries(predictions_path, output_dir)
    print("\nEvaluation pass finished.")
    print(f"Predictions : {predictions_path}")
    print(f"Summary     : {output_dir / 'summary.json'}")
    print(f"By language : {output_dir / 'summary_by_language.csv'}")
    print(f"By category: {output_dir / 'summary_by_category.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
