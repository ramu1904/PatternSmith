from collections import Counter
from pathlib import Path
import json
import random

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "NL-RX-Turk"

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits" / "NL-RX-Turk"

SOURCE_FILE = PROCESSED_DIR / "src.txt"
TARGET_FILE = PROCESSED_DIR / "targ.txt"

RANDOM_SEED = 42

TRAIN_RATIO = 0.65
VAL_RATIO = 0.10
TEST_RATIO = 0.25


# --------------------------------------------------
# 2. Load processed dataset
# --------------------------------------------------

descriptions = SOURCE_FILE.read_text(encoding="utf-8").splitlines()

targets = TARGET_FILE.read_text(encoding="utf-8").splitlines()

if len(descriptions) != len(targets):
    raise ValueError("Description and target counts do not match.")

if not descriptions:
    raise ValueError("Processed dataset is empty.")

total_examples = len(descriptions)


# --------------------------------------------------
# 3. Validate split configuration
# --------------------------------------------------

if not abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9:
    raise ValueError("Split ratios must sum to 1.0.")


# --------------------------------------------------
# 4. Audit repeated descriptions and targets
# --------------------------------------------------

description_counts = Counter(descriptions)
target_counts = Counter(targets)

repeated_descriptions = {
    description: count for description, count in description_counts.items() if count > 1
}

repeated_targets = {
    target: count for target, count in target_counts.items() if count > 1
}

# Our cleaning policy should guarantee unique descriptions.
if repeated_descriptions:
    raise ValueError(
        "Repeated descriptions found after cleaning. "
        "Investigate the cleaning pipeline before splitting."
    )


# --------------------------------------------------
# 5. Calculate exact split sizes
# --------------------------------------------------

ratios = [
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
]

raw_sizes = [total_examples * ratio for ratio in ratios]

split_sizes = [int(size) for size in raw_sizes]

remaining = total_examples - sum(split_sizes)

# Largest-remainder allocation ensures exact total.
fractional_parts = [size - int(size) for size in raw_sizes]

priority = sorted(
    range(len(ratios)),
    key=lambda index: fractional_parts[index],
    reverse=True,
)

for index in priority[:remaining]:
    split_sizes[index] += 1

train_size, val_size, test_size = split_sizes


# --------------------------------------------------
# 6. Deterministic random split
# --------------------------------------------------

indices = list(range(total_examples))

rng = random.Random(RANDOM_SEED)
rng.shuffle(indices)

train_indices = indices[:train_size]

val_indices = indices[train_size : train_size + val_size]

test_indices = indices[train_size + val_size :]


# --------------------------------------------------
# 7. Verify split integrity
# --------------------------------------------------

all_indices = train_indices + val_indices + test_indices

assert len(all_indices) == total_examples
assert len(set(all_indices)) == total_examples
assert set(all_indices) == set(range(total_examples))

train_descriptions = {descriptions[index] for index in train_indices}

val_descriptions = {descriptions[index] for index in val_indices}

test_descriptions = {descriptions[index] for index in test_indices}

assert train_descriptions.isdisjoint(val_descriptions)
assert train_descriptions.isdisjoint(test_descriptions)
assert val_descriptions.isdisjoint(test_descriptions)


# --------------------------------------------------
# 8. Analyze target overlap
# --------------------------------------------------

train_targets = {targets[index] for index in train_indices}

val_targets = {targets[index] for index in val_indices}

test_targets = {targets[index] for index in test_indices}


def calculate_target_overlap(split_indices, train_target_set):
    split_targets = [targets[index] for index in split_indices]

    seen_in_train = sum(target in train_target_set for target in split_targets)

    unseen_in_train = len(split_targets) - seen_in_train

    return {
        "examples": len(split_targets),
        "examples_with_target_seen_in_train": seen_in_train,
        "examples_with_unseen_target": unseen_in_train,
        "seen_target_percentage": round(
            seen_in_train / len(split_targets) * 100,
            2,
        ),
        "unseen_target_percentage": round(
            unseen_in_train / len(split_targets) * 100,
            2,
        ),
        "unique_targets": len(set(split_targets)),
        "unique_targets_also_in_train": len(set(split_targets) & train_target_set),
    }


target_overlap = {
    "validation": calculate_target_overlap(
        val_indices,
        train_targets,
    ),
    "test": calculate_target_overlap(
        test_indices,
        train_targets,
    ),
}


# --------------------------------------------------
# 9. Save split files
# --------------------------------------------------

split_mapping = {
    "train": train_indices,
    "validation": val_indices,
    "test": test_indices,
}

for split_name, split_indices in split_mapping.items():

    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    split_descriptions = [descriptions[index] for index in split_indices]

    split_targets = [targets[index] for index in split_indices]

    (split_dir / "src.txt").write_text(
        "\n".join(split_descriptions) + "\n",
        encoding="utf-8",
    )

    (split_dir / "targ.txt").write_text(
        "\n".join(split_targets) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------
# 10. Save reproducibility metadata
# --------------------------------------------------

metadata = {
    "dataset": "NL-RX-Turk",
    "total_examples": total_examples,
    "random_seed": RANDOM_SEED,
    "split_ratios": {
        "train": TRAIN_RATIO,
        "validation": VAL_RATIO,
        "test": TEST_RATIO,
    },
    "split_sizes": {
        "train": train_size,
        "validation": val_size,
        "test": test_size,
    },
    "description_audit": {
        "unique_descriptions": len(description_counts),
        "repeated_description_groups": len(repeated_descriptions),
    },
    "target_audit": {
        "unique_targets": len(target_counts),
        "repeated_target_groups": len(repeated_targets),
        "examples_with_repeated_targets": sum(
            target_counts[target] for target in repeated_targets
        ),
        "maximum_target_frequency": max(target_counts.values()),
    },
    "target_overlap_with_training": target_overlap,
    "policy": (
        "Deterministic random split after verifying that "
        "all descriptions are unique. Repeated regex targets "
        "are measured but not grouped across splits."
    ),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "split_indices.json").write_text(
    json.dumps(split_mapping, indent=4),
    encoding="utf-8",
)

(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=4),
    encoding="utf-8",
)


# --------------------------------------------------
# 11. Final verification
# --------------------------------------------------

for split_name, split_indices in split_mapping.items():

    split_dir = OUTPUT_DIR / split_name

    saved_descriptions = (
        (split_dir / "src.txt").read_text(encoding="utf-8").splitlines()
    )

    saved_targets = (split_dir / "targ.txt").read_text(encoding="utf-8").splitlines()

    assert len(saved_descriptions) == len(split_indices)
    assert len(saved_targets) == len(split_indices)

    assert len(saved_descriptions) == len(saved_targets)

assert train_descriptions.isdisjoint(val_descriptions)
assert train_descriptions.isdisjoint(test_descriptions)
assert val_descriptions.isdisjoint(test_descriptions)


# --------------------------------------------------
# 12. Print report
# --------------------------------------------------

print("\n========== DATASET SPLIT REPORT ==========")

print("Total examples:", total_examples)
print("Random seed:", RANDOM_SEED)

print("\nSplit sizes:")
print("Training:", train_size)
print("Validation:", val_size)
print("Testing:", test_size)

print("\nDescription audit:")
print("Unique descriptions:", len(description_counts))
print("Repeated description groups:", len(repeated_descriptions))

print("\nTarget audit:")
print("Unique targets:", len(target_counts))
print("Repeated target groups:", len(repeated_targets))
print(
    "Examples belonging to repeated target groups:",
    metadata["target_audit"]["examples_with_repeated_targets"],
)
print(
    "Maximum frequency of one target:",
    metadata["target_audit"]["maximum_target_frequency"],
)

print("\nTarget overlap with training:")

for split_name, result in target_overlap.items():

    print(f"\n{split_name.upper()}")

    for key, value in result.items():
        print(f"{key}: {value}")

print("\nOutput directory:", OUTPUT_DIR)

print("\nDataset splitting and verification completed.")
