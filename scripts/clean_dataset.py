from collections import defaultdict
from pathlib import Path
import csv
import json

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "NL-RX-Turk"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "NL-RX-Turk"

SOURCE_FILE = RAW_DIR / "src.txt"
TARGET_FILE = RAW_DIR / "targ.txt"


# --------------------------------------------------
# 2. Load original data
# --------------------------------------------------

descriptions = SOURCE_FILE.read_text(encoding="utf-8").splitlines()
targets = TARGET_FILE.read_text(encoding="utf-8").splitlines()

if len(descriptions) != len(targets):
    raise ValueError("Source and target counts do not match.")

original_count = len(descriptions)
pairs = list(zip(descriptions, targets))


# --------------------------------------------------
# 3. Identify conflicting descriptions
# --------------------------------------------------

description_to_targets = defaultdict(set)

for description, target in pairs:
    description_to_targets[description].add(target)

conflicting_descriptions = {
    description
    for description, distinct_targets in description_to_targets.items()
    if len(distinct_targets) > 1
}


# --------------------------------------------------
# 4. Clean data and record exclusions
# --------------------------------------------------

unique_pairs = []
seen_pairs = set()
excluded_rows = []

for index, (description, target) in enumerate(pairs):

    pair = (description, target)

    # Exclusion rule 1: conflicting descriptions
    if description in conflicting_descriptions:
        reason = "conflicting_description"

    # Exclusion rule 2: repeated exact pairs
    elif pair in seen_pairs:
        reason = "duplicate_pair"

    # Retain the first occurrence
    else:
        seen_pairs.add(pair)
        unique_pairs.append(pair)
        continue

    excluded_rows.append(
        {
            "original_index": index,
            "description": description,
            "target": target,
            "reason": reason,
        }
    )


# --------------------------------------------------
# 5. Save processed dataset
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

clean_descriptions = [description for description, _ in unique_pairs]
clean_targets = [target for _, target in unique_pairs]

(OUTPUT_DIR / "src.txt").write_text(
    "\n".join(clean_descriptions) + "\n",
    encoding="utf-8",
)

(OUTPUT_DIR / "targ.txt").write_text(
    "\n".join(clean_targets) + "\n",
    encoding="utf-8",
)


# --------------------------------------------------
# 6. Save exclusion report
# --------------------------------------------------

with (OUTPUT_DIR / "excluded_examples.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "original_index",
            "description",
            "target",
            "reason",
        ],
    )

    writer.writeheader()
    writer.writerows(excluded_rows)


# --------------------------------------------------
# 7. Save preprocessing metadata
# --------------------------------------------------

conflicting_rows = sum(
    description in conflicting_descriptions for description, _ in pairs
)

duplicate_rows_removed = sum(row["reason"] == "duplicate_pair" for row in excluded_rows)

metadata = {
    "original_examples": original_count,
    "conflicting_descriptions": len(conflicting_descriptions),
    "rows_with_conflicting_descriptions": conflicting_rows,
    "exact_duplicate_pairs_removed": duplicate_rows_removed,
    "processed_examples": len(unique_pairs),
    "excluded_examples": len(excluded_rows),
    "policy": (
        "Exclude descriptions with multiple distinct target strings; "
        "remove subsequent occurrences of exact duplicate pairs."
    ),
}


(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=4),
    encoding="utf-8",
)


# --------------------------------------------------
# 8. Final verification
# --------------------------------------------------

saved_descriptions = (OUTPUT_DIR / "src.txt").read_text(encoding="utf-8").splitlines()

saved_targets = (OUTPUT_DIR / "targ.txt").read_text(encoding="utf-8").splitlines()

saved_pairs = list(zip(saved_descriptions, saved_targets))

assert len(saved_descriptions) == len(saved_targets)

assert len(saved_pairs) == len(set(saved_pairs))

assert not (set(saved_descriptions) & conflicting_descriptions)

assert len(saved_pairs) + len(excluded_rows) == original_count

assert len(excluded_rows) == (conflicting_rows + duplicate_rows_removed)


# --------------------------------------------------
# 9. Cleaning report
# --------------------------------------------------

print("\n========== CLEANING REPORT ==========")

for key, value in metadata.items():
    print(f"{key}: {value}")

print("\nExclusion breakdown:")

print(f"Conflicting-description rows: {conflicting_rows}")
print(f"Duplicate rows: {duplicate_rows_removed}")

print("\nOutput directory:", OUTPUT_DIR)

print("\nDataset cleaning and verification completed.")
