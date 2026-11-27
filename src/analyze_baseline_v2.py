
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path("/data/sriparna/arnav/LLM Security")
INPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR = INPUT_DIR / "baseline_analysis"

LABELS = ["Safe", "Unsafe", "Controversial"]


def calculate_metrics(df, group_name):
    """Calculate precision, recall and F1 for each label."""

    results = []

    for label in LABELS:
        tp = ((df["expected_label"] == label) &
              (df["predicted_label"] == label)).sum()

        fp = ((df["expected_label"] != label) &
              (df["predicted_label"] == label)).sum()

        fn = ((df["expected_label"] == label) &
              (df["predicted_label"] != label)).sum()

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0
        )

        support = int((df["expected_label"] == label).sum())

        results.append({
            "group": group_name,
            "label": label,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        })

    return results


def main():

    # Find the most recent baseline CSV
    csv_files = list(INPUT_DIR.glob("baseline_v2_results_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No baseline_v2_results_*.csv found in {INPUT_DIR}"
        )

    input_file = max(csv_files, key=lambda p: p.stat().st_mtime)

    print(f"\nReading: {input_file.name}")

    df = pd.read_csv(input_file)

    required_columns = [
        "semantic_group",
        "language",
        "intent",
        "expected_label",
        "predicted_label"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Standardise label formatting
    for col in ["expected_label", "predicted_label"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    df = df[
        df["expected_label"].isin(LABELS) &
        df["predicted_label"].isin(LABELS)
    ].copy()

    if df.empty:
        raise ValueError("No valid labelled records found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --------------------------------------------------
    # 1. Overall confusion matrix
    # --------------------------------------------------

    confusion = pd.crosstab(
        df["expected_label"],
        df["predicted_label"]
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)

    confusion.index.name = "Expected"
    confusion.columns.name = "Predicted"

    accuracy = (
        (df["expected_label"] == df["predicted_label"]).mean()
    )

    metric_rows = calculate_metrics(df, "Overall")
    macro_f1 = sum(x["f1_score"] for x in metric_rows) / len(LABELS)
    balanced_accuracy = sum(x["recall"] for x in metric_rows) / len(LABELS)

    # --------------------------------------------------
    # 2. Per-language metrics
    # --------------------------------------------------

    language_rows = []

    for language, subset in df.groupby("language"):
        language_rows.extend(calculate_metrics(subset, language))

    language_metrics = pd.DataFrame(language_rows)

    language_accuracy = (
        df.assign(
            correct=df["expected_label"] == df["predicted_label"]
        )
        .groupby("language")["correct"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index(name="agreement_percent")
    )

    # --------------------------------------------------
    # 3. Per-intent agreement
    # --------------------------------------------------

    intent_metrics = (
        df.assign(
            correct=df["expected_label"] == df["predicted_label"]
        )
        .groupby("intent")
        .agg(
            total=("correct", "size"),
            correct=("correct", "sum")
        )
        .reset_index()
    )

    intent_metrics["agreement_percent"] = (
        intent_metrics["correct"] /
        intent_metrics["total"] * 100
    ).round(2)

    # --------------------------------------------------
    # 4. Error analysis
    # --------------------------------------------------

    errors = df[
        df["expected_label"] != df["predicted_label"]
    ].copy()

    false_safe = df[
        (df["expected_label"] == "Unsafe") &
        (df["predicted_label"] == "Safe")
    ].copy()

    overblocking = df[
        (df["expected_label"] == "Safe") &
        (df["predicted_label"] == "Unsafe")
    ].copy()

    controversial_mismatches = df[
        (df["expected_label"] == "Controversial") &
        (df["predicted_label"] != "Controversial")
    ].copy()

    # --------------------------------------------------
    # 5. Semantic-group consistency
    # --------------------------------------------------

    consistency_rows = []

    for group, subset in df.groupby("semantic_group"):

        predictions = subset["predicted_label"].tolist()
        unique_predictions = sorted(set(predictions))

        consistency_rows.append({
            "semantic_group": group,
            "languages_evaluated": len(subset),
            "predicted_labels": " | ".join(predictions),
            "unique_prediction_count": len(unique_predictions),
            "consistent_across_languages": len(unique_predictions) == 1
        })

    consistency = pd.DataFrame(consistency_rows)

    consistent_groups = int(
        consistency["consistent_across_languages"].sum()
    )

    total_groups = len(consistency)

    # --------------------------------------------------
    # 6. Save output files
    # --------------------------------------------------

    confusion.to_csv(
        OUTPUT_DIR / f"confusion_matrix_{timestamp}.csv"
    )

    language_metrics.to_csv(
        OUTPUT_DIR / f"language_metrics_{timestamp}.csv",
        index=False
    )

    language_accuracy.to_csv(
        OUTPUT_DIR / f"language_agreement_{timestamp}.csv",
        index=False
    )

    intent_metrics.to_csv(
        OUTPUT_DIR / f"intent_analysis_{timestamp}.csv",
        index=False
    )

    errors.to_csv(
        OUTPUT_DIR / f"all_errors_{timestamp}.csv",
        index=False
    )

    false_safe.to_csv(
        OUTPUT_DIR / f"false_safe_cases_{timestamp}.csv",
        index=False
    )

    overblocking.to_csv(
        OUTPUT_DIR / f"overblocking_cases_{timestamp}.csv",
        index=False
    )

    controversial_mismatches.to_csv(
        OUTPUT_DIR / f"controversial_mismatches_{timestamp}.csv",
        index=False
    )

    consistency.to_csv(
        OUTPUT_DIR / f"semantic_group_consistency_{timestamp}.csv",
        index=False
    )

    summary = {
        "input_file": input_file.name,
        "total_records": int(len(df)),
        "overall_accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "total_errors": int(len(errors)),
        "false_safe_cases": int(len(false_safe)),
        "overblocking_cases": int(len(overblocking)),
        "controversial_mismatches": int(len(controversial_mismatches)),
        "consistent_semantic_groups": consistent_groups,
        "total_semantic_groups": total_groups
    }

    with open(
        OUTPUT_DIR / f"summary_{timestamp}.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------
    # 7. Generate readable Markdown report
    # --------------------------------------------------

    report = [
        "# Baseline V2 Analysis",
        "",
        f"Input file: `{input_file.name}`",
        "",
        "## Overall results",
        "",
        f"- Records analysed: {len(df)}",
        f"- Accuracy: {accuracy * 100:.2f}%",
        f"- Macro F1: {macro_f1:.4f}",
        f"- Balanced accuracy: {balanced_accuracy * 100:.2f}%",
        f"- Total mismatches: {len(errors)}",
        f"- False-safe cases: {len(false_safe)}",
        f"- Overblocking cases: {len(overblocking)}",
        f"- Controversial mismatches: {len(controversial_mismatches)}",
        "",
        "## Confusion matrix",
        "",
        confusion.to_markdown(),
        "",
        "Rows = expected labels; columns = predicted labels.",
        "",
        "## Per-language agreement",
        "",
        language_accuracy.to_markdown(index=False),
        "",
        "## Per-language classification metrics",
        "",
        language_metrics.to_markdown(index=False),
        "",
        "## Per-intent agreement",
        "",
        intent_metrics.to_markdown(index=False),
        "",
        "## Semantic-group consistency",
        "",
        f"- Consistent groups: {consistent_groups}/{total_groups}",
        "",
        consistency.to_markdown(index=False),
        "",
        "## Interpretation caution",
        "",
        "These results measure agreement with provisional expected labels. "
        "They should not be presented as definitive model accuracy until "
        "the labels and translations have been human-reviewed.",
        "",
        "The 100 records represent 25 semantic groups across four language "
        "representations, not 100 independent scenarios."
    ]

    report_file = OUTPUT_DIR / f"analysis_report_{timestamp}.md"
    report_file.write_text("\n".join(report), encoding="utf-8")

    # --------------------------------------------------
    # 8. Print key results
    # --------------------------------------------------

    print("\n========== BASELINE ANALYSIS ==========")
    print(f"Records analysed: {len(df)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"Total mismatches: {len(errors)}")
    print(f"False-safe cases: {len(false_safe)}")
    print(f"Overblocking cases: {len(overblocking)}")
    print(f"Controversial mismatches: {len(controversial_mismatches)}")

    print("\nConfusion Matrix:")
    print(confusion)

    print("\nLanguage Agreement:")
    print(language_accuracy.to_string(index=False))

    print("\nIntent Agreement:")
    print(intent_metrics.to_string(index=False))

    print("\nSemantic-group consistency:")
    print(f"{consistent_groups}/{total_groups} groups")

    print(f"\nAll files saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
