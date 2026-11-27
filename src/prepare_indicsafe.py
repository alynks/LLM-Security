#!/usr/bin/env python3
"""Prepare the IndicSafe benchmark for multilingual safety-guard experiments.

What this script does
---------------------
1. Loads ppattnay/IndicSafe (benchmark split) from the Hugging Face Datasets cache/Hub.
2. Preserves all 6,000 original translated rows and their original fields.
3. Creates one English source record for each unique normalised English prompt.
4. Creates stable semantic-group IDs from the English source prompt. All language
   variants and duplicate source rows therefore remain in the same split.
5. Normalises the known category alias "Harmless Control" to
   "Harmless Control Prompts", while preserving both original category columns.
6. Produces group-wise, category-stratified calibration/evaluation splits.
7. Writes JSONL, CSV, audit files, and a manifest.

No source records are deleted. The translated-record file retains duplicate rows;
identical English source strings are represented once in english_records.jsonl.

Example:
    python src/prepare_indicsafe.py --output-dir datasets/processed/indicsafe_v1

Dependencies: `datasets` (the script uses Python standard-library CSV/JSON tools).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_REPO_ID = "ppattnay/IndicSafe"
DEFAULT_SPLIT = "benchmark"
REQUIRED_COLUMNS = ["Prompt", "Category", "Language", "Translated_Prompt", "category"]

# The inspection output showed these 10 raw labels. Only the two Harmless Control
# labels are aliases for the same paper-level category. All original labels are
# retained in `raw_category` and `raw_category_legacy`.
CATEGORY_ALIASES = {
    "harmless control": "Harmless Control Prompts",
    "harmless control prompts": "Harmless Control Prompts",
}

LANGUAGE_CODES = {
    "bengali": "bn",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "nepali": "ne",
    "odia": "or",
    "oriya": "or",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
    "english": "en",
}


def clean_text(value: Any) -> str:
    """Return a display-safe string without changing its substantive content."""
    if value is None:
        return ""
    return str(value).strip()


def normalise_for_grouping(text: str) -> str:
    """Normalise English text only for duplicate detection and stable IDs."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def stable_hash(text: str, length: int = 12) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest().upper()
    return digest[:length]


def make_group_id(prompt: str) -> str:
    normalised = normalise_for_grouping(prompt)
    return "IS-" + stable_hash(normalised)


def canonicalise_category(raw_category: str) -> str:
    key = re.sub(r"\s+", " ", raw_category).strip().casefold()
    return CATEGORY_ALIASES.get(key, raw_category.strip())


def language_code(language: str) -> str:
    key = language.strip().casefold()
    if key in LANGUAGE_CODES:
        return LANGUAGE_CODES[key]
    # Preserve unknown languages rather than failing or silently relabelling them.
    slug = re.sub(r"[^a-z0-9]+", "", key)
    return (slug[:8] or "unknown")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def stratified_group_split(
    group_rows: Sequence[Mapping[str, Any]],
    calibration_fraction: float,
    seed: int,
) -> Dict[str, str]:
    """Create a deterministic category-stratified split at semantic-group level."""
    by_category: Dict[str, List[str]] = defaultdict(list)
    for group in group_rows:
        cats = group.get("canonical_categories", [])
        if len(cats) == 1:
            stratum = cats[0]
        elif len(cats) == 0:
            stratum = "CATEGORY_MISSING"
        else:
            stratum = "CATEGORY_CONFLICT"
        by_category[stratum].append(str(group["seed_group_id"]))

    rng = random.Random(seed)
    assignments: Dict[str, str] = {}
    for category in sorted(by_category):
        ids = sorted(by_category[category])
        rng.shuffle(ids)
        n = len(ids)
        if n <= 1:
            n_eval = 0
        else:
            n_eval = int(math.floor(n * (1.0 - calibration_fraction) + 0.5))
            n_eval = max(1, min(n - 1, n_eval))
        evaluation_ids = set(ids[:n_eval])
        for group_id in ids:
            assignments[group_id] = "evaluation" if group_id in evaluation_ids else "calibration"
    return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare IndicSafe records, semantic groups, category aliases, and group-wise splits."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository ID")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to load")
    parser.add_argument(
        "--output-dir",
        default="datasets/processed/indicsafe_v1",
        help="Output directory (relative to current project directory by default)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group split")
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.60,
        help="Fraction of semantic groups assigned to calibration (default: 0.60)",
    )
    parser.add_argument(
        "--expected-languages",
        type=int,
        default=12,
        help="Expected number of translated languages; warning only (default: 12)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        print("ERROR: --calibration-fraction must be between 0 and 1.", file=sys.stderr)
        return 2

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: The 'datasets' package is missing from the Python environment.\n"
            "Install it in the active environment with: python -m pip install datasets",
            file=sys.stderr,
        )
        return 2

    print(f"Loading {args.repo_id} (split={args.split!r}) ...")
    try:
        dataset = load_dataset(args.repo_id, split=args.split)
    except Exception as exc:  # give the user a useful error without hiding the cause
        print(f"ERROR: Could not load dataset: {exc}", file=sys.stderr)
        print("Check the repository ID, split name, network access, and Hugging Face cache.", file=sys.stderr)
        return 1

    columns = list(dataset.column_names)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        print(f"ERROR: Dataset is missing required columns: {missing_columns}", file=sys.stderr)
        print(f"Available columns: {columns}", file=sys.stderr)
        return 1

    raw_rows: List[Dict[str, Any]] = []
    groups: Dict[str, Dict[str, Any]] = {}
    translated_rows: List[Dict[str, Any]] = []
    category_raw_counts: Counter[str] = Counter()
    category_canonical_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    legacy_category_mismatches = 0
    missing_required_values = Counter()

    for index, item in enumerate(dataset):
        prompt = clean_text(item.get("Prompt"))
        translated_prompt = clean_text(item.get("Translated_Prompt"))
        category_raw = clean_text(item.get("Category"))
        legacy_category = clean_text(item.get("category"))
        language = clean_text(item.get("Language"))

        for name, value in (
            ("Prompt", prompt),
            ("Translated_Prompt", translated_prompt),
            ("Category", category_raw),
            ("Language", language),
        ):
            if not value:
                missing_required_values[name] += 1

        if prompt and translated_prompt and category_raw and language:
            canonical_category = canonicalise_category(category_raw)
            group_id = make_group_id(prompt)
            lang_code = language_code(language)
        else:
            # Preserve incomplete rows in the raw export, but do not silently create
            # malformed benchmark records from them.
            canonical_category = canonicalise_category(category_raw) if category_raw else ""
            group_id = make_group_id(prompt) if prompt else f"IS-MISSING-{index:06d}"
            lang_code = language_code(language) if language else "unknown"

        if legacy_category and category_raw and legacy_category.strip() != category_raw.strip():
            legacy_category_mismatches += 1

        raw_record = {
            "source_row_index": index,
            "Prompt": prompt,
            "Category": category_raw,
            "Language": language,
            "Translated_Prompt": translated_prompt,
            "category": legacy_category,
            "seed_group_id": group_id,
            "canonical_category": canonical_category,
        }
        raw_rows.append(raw_record)
        category_raw_counts[category_raw or "__MISSING__"] += 1
        category_canonical_counts[canonical_category or "__MISSING__"] += 1
        language_counts[language or "__MISSING__"] += 1

        if not (prompt and translated_prompt and category_raw and language):
            continue

        group = groups.setdefault(
            group_id,
            {
                "seed_group_id": group_id,
                "seed_prompt_en": prompt,
                "normalised_seed_prompt": normalise_for_grouping(prompt),
                "source_row_indices": [],
                "raw_categories": set(),
                "canonical_categories": set(),
                "translations_by_language": defaultdict(set),
            },
        )
        group["source_row_indices"].append(index)
        group["raw_categories"].add(category_raw)
        group["canonical_categories"].add(canonical_category)
        group["translations_by_language"][language].add(translated_prompt)

        translated_rows.append(
            {
                "record_id": f"{group_id}-{lang_code.upper()}-{index:05d}",
                "record_type": "translated_prompt",
                "source_row_index": index,
                "seed_group_id": group_id,
                "prompt": translated_prompt,
                "prompt_en": prompt,
                "language": language,
                "language_code": lang_code,
                "script": "native",
                "raw_category": category_raw,
                "legacy_category": legacy_category,
                "canonical_category": canonical_category,
                "category_conflict_in_seed_group": False,  # updated after grouping
                "source_dataset": args.repo_id,
                "source_split": args.split,
            }
        )

    if not groups:
        print("ERROR: No complete benchmark rows were found.", file=sys.stderr)
        return 1

    group_rows: List[Dict[str, Any]] = []
    duplicate_audit: List[Dict[str, Any]] = []
    english_rows: List[Dict[str, Any]] = []

    for group_id in sorted(groups):
        group = groups[group_id]
        raw_categories = sorted(group["raw_categories"])
        canonical_categories = sorted(group["canonical_categories"])
        translations = {
            lang: sorted(texts)
            for lang, texts in sorted(group["translations_by_language"].items())
        }
        category_conflict = len(canonical_categories) > 1
        group_row = {
            "seed_group_id": group_id,
            "seed_prompt_en": group["seed_prompt_en"],
            "source_row_count": len(group["source_row_indices"]),
            "source_row_indices": sorted(group["source_row_indices"]),
            "raw_categories": raw_categories,
            "canonical_categories": canonical_categories,
            "canonical_category": canonical_categories[0] if len(canonical_categories) == 1 else None,
            "category_conflict": category_conflict,
            "language_count": len(translations),
            "translations_by_language": translations,
            "translation_variant_count": sum(len(values) for values in translations.values()),
        }
        group_rows.append(group_row)

        if len(group["source_row_indices"]) > 1 or category_conflict:
            duplicate_audit.append(
                {
                    "seed_group_id": group_id,
                    "seed_prompt_en": group["seed_prompt_en"],
                    "source_row_count": len(group["source_row_indices"]),
                    "source_row_indices": sorted(group["source_row_indices"]),
                    "raw_categories": raw_categories,
                    "canonical_categories": canonical_categories,
                    "category_conflict": category_conflict,
                    "language_count": len(translations),
                    "translation_variant_count": sum(len(values) for values in translations.values()),
                    "translations_by_language": translations,
                }
            )

        english_rows.append(
            {
                "record_id": f"{group_id}-EN",
                "record_type": "english_source_prompt",
                "source_row_index": min(group["source_row_indices"]),
                "seed_group_id": group_id,
                "prompt": group["seed_prompt_en"],
                "prompt_en": group["seed_prompt_en"],
                "language": "English",
                "language_code": "en",
                "script": "Latin",
                "raw_category": raw_categories[0] if len(raw_categories) == 1 else None,
                "raw_categories": raw_categories,
                "canonical_category": canonical_categories[0] if len(canonical_categories) == 1 else None,
                "canonical_categories": canonical_categories,
                "category_conflict_in_seed_group": category_conflict,
                "source_row_indices": sorted(group["source_row_indices"]),
                "source_dataset": args.repo_id,
                "source_split": args.split,
            }
        )

    split_assignments = stratified_group_split(
        group_rows=group_rows,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )

    for row in translated_rows:
        row["split"] = split_assignments[row["seed_group_id"]]
        # Check whether this source prompt has conflicting categories across rows.
        row["category_conflict_in_seed_group"] = len(
            groups[row["seed_group_id"]]["canonical_categories"]
        ) > 1

    for row in english_rows:
        row["split"] = split_assignments[row["seed_group_id"]]

    for row in group_rows:
        row["split"] = split_assignments[row["seed_group_id"]]

    all_records = english_rows + translated_rows
    calibration_records = [row for row in all_records if row["split"] == "calibration"]
    evaluation_records = [row for row in all_records if row["split"] == "evaluation"]

    split_rows = []
    for group in group_rows:
        split_rows.append(
            {
                "seed_group_id": group["seed_group_id"],
                "seed_prompt_en": group["seed_prompt_en"],
                "canonical_category": group["canonical_category"],
                "canonical_categories": group["canonical_categories"],
                "category_conflict": group["category_conflict"],
                "source_row_count": group["source_row_count"],
                "split": group["split"],
            }
        )

    # Distribution summaries: count original translated rows and unique source groups.
    language_summary = [
        {"language": language, "translated_record_count": count}
        for language, count in sorted(language_counts.items())
    ]
    category_summary = []
    for category in sorted(set(category_raw_counts) | set(category_canonical_counts)):
        category_summary.append(
            {
                "category_label": category,
                "raw_row_count": category_raw_counts.get(category, 0),
                "canonical_row_count": category_canonical_counts.get(category, 0),
            }
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve an auditable copy of the original 6,000-row schema plus derived IDs.
    raw_fields = [
        "source_row_index", "Prompt", "Category", "Language", "Translated_Prompt",
        "category", "seed_group_id", "canonical_category",
    ]
    raw_jsonl_count = write_jsonl(output_dir / "raw_rows_with_ids.jsonl", raw_rows)
    write_csv(output_dir / "raw_rows_with_ids.csv", raw_rows, raw_fields)

    translated_fields = [
        "record_id", "record_type", "source_row_index", "seed_group_id", "prompt", "prompt_en",
        "language", "language_code", "script", "raw_category", "legacy_category",
        "canonical_category", "category_conflict_in_seed_group", "split", "source_dataset", "source_split",
    ]
    english_fields = [
        "record_id", "record_type", "source_row_index", "seed_group_id", "prompt", "prompt_en",
        "language", "language_code", "script", "raw_category", "raw_categories",
        "canonical_category", "canonical_categories", "category_conflict_in_seed_group",
        "source_row_indices", "split", "source_dataset", "source_split",
    ]
    group_fields = [
        "seed_group_id", "seed_prompt_en", "source_row_count", "source_row_indices",
        "raw_categories", "canonical_categories", "canonical_category", "category_conflict",
        "language_count", "translations_by_language", "translation_variant_count", "split",
    ]
    all_fields = sorted(set(translated_fields) | set(english_fields))

    write_jsonl(output_dir / "translated_records.jsonl", translated_rows)
    write_csv(output_dir / "translated_records.csv", translated_rows, translated_fields)
    write_jsonl(output_dir / "english_records.jsonl", english_rows)
    write_csv(output_dir / "english_records.csv", english_rows, english_fields)
    write_jsonl(output_dir / "seed_groups.jsonl", group_rows)
    write_csv(output_dir / "seed_groups.csv", group_rows, group_fields)
    write_jsonl(output_dir / "all_records.jsonl", all_records)
    write_csv(output_dir / "all_records.csv", all_records, all_fields)
    write_jsonl(output_dir / "calibration_records.jsonl", calibration_records)
    write_jsonl(output_dir / "evaluation_records.jsonl", evaluation_records)
    write_csv(output_dir / "seed_splits.csv", split_rows, list(split_rows[0].keys()))
    write_csv(
        output_dir / "duplicate_and_conflict_audit.csv",
        duplicate_audit,
        [
            "seed_group_id", "seed_prompt_en", "source_row_count", "source_row_indices",
            "raw_categories", "canonical_categories", "category_conflict", "language_count",
            "translation_variant_count", "translations_by_language",
        ],
    )
    write_csv(output_dir / "language_distribution.csv", language_summary, ["language", "translated_record_count"])
    write_csv(
        output_dir / "category_distribution.csv",
        category_summary,
        ["category_label", "raw_row_count", "canonical_row_count"],
    )

    canonical_category_counts = Counter(row["canonical_category"] for row in translated_rows)
    raw_language_names = sorted(language_counts)
    split_group_counts = Counter(split_assignments.values())
    split_record_counts = Counter(row["split"] for row in all_records)
    category_conflict_groups = [row["seed_group_id"] for row in group_rows if row["category_conflict"]]
    category_mapped_rows = sum(
        1 for row in raw_rows if row["Category"] and canonicalise_category(row["Category"]) != row["Category"].strip()
    )

    manifest = {
        "dataset_repo_id": args.repo_id,
        "dataset_split": args.split,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_rows": len(raw_rows),
        "translated_record_count": len(translated_rows),
        "unique_english_source_record_count": len(english_rows),
        "combined_record_count": len(all_records),
        "unique_semantic_group_count": len(group_rows),
        "raw_category_count": len(category_raw_counts),
        "canonical_category_count": len(set(canonical_category_counts)),
        "raw_category_distribution": dict(sorted(category_raw_counts.items())),
        "canonical_category_distribution_translated_rows": dict(sorted(canonical_category_counts.items())),
        "language_distribution": dict(sorted(language_counts.items())),
        "source_prompt_duplicates_grouped": sum(1 for group in group_rows if group["source_row_count"] > 1),
        "category_conflict_group_count": len(category_conflict_groups),
        "category_conflict_group_ids": category_conflict_groups,
        "category_alias_mapped_row_count": category_mapped_rows,
        "legacy_category_mismatch_row_count": legacy_category_mismatches,
        "missing_required_values": dict(missing_required_values),
        "split": {
            "method": "category-stratified semantic-group split",
            "calibration_fraction_requested": args.calibration_fraction,
            "evaluation_fraction_requested": 1.0 - args.calibration_fraction,
            "random_seed": args.seed,
            "semantic_group_counts": dict(split_group_counts),
            "record_counts": dict(split_record_counts),
        },
        "language_names_observed": raw_language_names,
        "expected_language_count": args.expected_languages,
        "outputs": [
            "raw_rows_with_ids.jsonl", "raw_rows_with_ids.csv",
            "translated_records.jsonl", "translated_records.csv",
            "english_records.jsonl", "english_records.csv",
            "seed_groups.jsonl", "seed_groups.csv",
            "all_records.jsonl", "all_records.csv",
            "calibration_records.jsonl", "evaluation_records.jsonl",
            "seed_splits.csv", "duplicate_and_conflict_audit.csv",
            "language_distribution.csv", "category_distribution.csv",
        ],
        "notes": [
            "All original translated rows are preserved; duplicate source prompts are grouped for splitting.",
            "English source records are deduplicated by Unicode-normalised, whitespace-normalised, case-folded prompt text.",
            "The original Category and category fields are preserved in raw_rows_with_ids files.",
            "Only the known Harmless Control alias is normalised to Harmless Control Prompts.",
            "Category conflicts within a semantic group are flagged and must be reviewed before using that group as gold-labelled evaluation data.",
            "This script does not create Romanized or code-mixed variants and does not claim that translations are human-validated.",
            "The split is at semantic-group level so all language variants of a seed prompt remain in one partition.",
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    readme = f"""# IndicSafe prepared benchmark

- Source: `{args.repo_id}` / split `{args.split}`
- Original translated rows: **{len(raw_rows):,}**
- Unique English source records: **{len(english_rows):,}**
- Combined records: **{len(all_records):,}**
- Semantic groups: **{len(group_rows):,}**
- Canonical categories: **{len(set(canonical_category_counts))}**

## Important files

- `seed_groups.jsonl`: one record per unique English seed prompt, with translations grouped by language.
- `translated_records.jsonl`: all original translated rows, preserving duplicates and source-row indices.
- `english_records.jsonl`: one English source record per unique seed prompt.
- `all_records.jsonl`: English source records plus all translated records.
- `seed_splits.csv`: calibration/evaluation assignment by semantic group.
- `duplicate_and_conflict_audit.csv`: repeated source prompts and category/translation variation.
- `manifest.json`: dataset counts, split settings and provenance.

## Important cautions

1. The source benchmark's original category is not automatically a human-verified Safe/Unsafe guard label. Define and validate a separate guard-evaluation gold label before calculating FNR/FPR or classification accuracy.
2. Review every category-conflict group before using it as a gold-labelled evaluation item.
3. Translation availability is not equivalent to human validation. Human reviewers should check the language and semantic equivalence of the evaluation subset.
4. All variants sharing a seed group ID receive the same split to prevent cross-language leakage.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print("\n" + "=" * 72)
    print("INDICSAFE PREPROCESSING COMPLETE")
    print("=" * 72)
    print(f"Original dataset rows          : {len(raw_rows):,}")
    print(f"Translated records preserved   : {len(translated_rows):,}")
    print(f"Unique English source records  : {len(english_rows):,}")
    print(f"Combined records               : {len(all_records):,}")
    print(f"Unique semantic groups         : {len(group_rows):,}")
    print(f"Raw category labels            : {len(category_raw_counts)}")
    print(f"Canonical category labels      : {len(set(canonical_category_counts))}")
    print(f"Category-alias rows normalised  : {category_mapped_rows:,}")
    print(f"Category-conflict groups       : {len(category_conflict_groups):,}")
    print(f"Legacy category mismatches     : {legacy_category_mismatches:,}")
    print(f"Calibration groups             : {split_group_counts.get('calibration', 0):,}")
    print(f"Evaluation groups              : {split_group_counts.get('evaluation', 0):,}")
    print(f"Calibration records            : {len(calibration_records):,}")
    print(f"Evaluation records             : {len(evaluation_records):,}")
    print(f"Observed languages              : {len(language_counts)}")
    if len(language_counts) != args.expected_languages:
        print(
            f"WARNING: Expected {args.expected_languages} languages but found {len(language_counts)}. "
            "Review language_distribution.csv."
        )
    if len(set(canonical_category_counts)) != 9:
        print("WARNING: Expected 9 canonical categories after alias normalisation; review category_distribution.csv.")
    if category_conflict_groups:
        print("WARNING: Some seed groups contain conflicting categories. Review duplicate_and_conflict_audit.csv.")
    if missing_required_values:
        print(f"WARNING: Missing required values found: {dict(missing_required_values)}")
    print(f"\nOutput directory: {output_dir}")
    print("See manifest.json and README.md for full provenance and notes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
