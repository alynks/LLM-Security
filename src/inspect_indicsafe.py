from datasets import load_dataset

# Load the official IndicSafe benchmark
dataset = load_dataset(
    "ppattnay/IndicSafe",
    split="benchmark"
)

print("Total records:", len(dataset))
print("Columns:", dataset.column_names)

# Display first three records
for i in range(3):
    print(f"\nRecord {i}")
    print(dataset[i])

# Language distribution
df = dataset.to_pandas()

print("\nLanguage distribution:")
print(df["Language"].value_counts())

# Canonical category distribution
print("\nCategory distribution:")
print(df["Category"].value_counts())

print("\nUnique English prompts:",
      df["Prompt"].nunique())

print("\nMissing translations:",
      df["Translated_Prompt"].isna().sum())