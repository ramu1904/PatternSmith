from collections import Counter, defaultdict
from pathlib import Path
import statistics

# --------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "NL-RX-Turk"

SOURCE_FILE = DATA_DIR / "src.txt"
TARGET_FILE = DATA_DIR / "targ.txt"


# --------------------------------------------------
# 2. Helper function: calculate percentiles
# --------------------------------------------------


def percentile(values, percentage):
    """
    Calculate a percentile using linear interpolation.
    """
    if not values:
        return 0

    sorted_values = sorted(values)

    position = (len(sorted_values) - 1) * percentage / 100

    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)

    weight = position - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------

if not SOURCE_FILE.exists():
    raise FileNotFoundError(f"Missing file: {SOURCE_FILE}")

if not TARGET_FILE.exists():
    raise FileNotFoundError(f"Missing file: {TARGET_FILE}")

descriptions = SOURCE_FILE.read_text(encoding="utf-8").splitlines()

regex_targets = TARGET_FILE.read_text(encoding="utf-8").splitlines()


# --------------------------------------------------
# 4. Validate dataset alignment
# --------------------------------------------------

print("\n========== DATASET OVERVIEW ==========")

print("Descriptions:", len(descriptions))
print("Regex targets:", len(regex_targets))

if len(descriptions) != len(regex_targets):
    raise ValueError("Dataset alignment error: source and target counts differ.")

total_examples = len(descriptions)

print("Total paired examples:", total_examples)


# --------------------------------------------------
# 5. Check empty and whitespace records
# --------------------------------------------------

empty_descriptions = [i for i, item in enumerate(descriptions) if not item.strip()]

empty_targets = [i for i, item in enumerate(regex_targets) if not item.strip()]

whitespace_issues = [
    i
    for i, (src, tgt) in enumerate(zip(descriptions, regex_targets))
    if src != src.strip() or tgt != tgt.strip()
]

print("\n========== DATA QUALITY ==========")

print("Empty descriptions:", len(empty_descriptions))
print("Empty regex targets:", len(empty_targets))
print("Whitespace issues:", len(whitespace_issues))


# --------------------------------------------------
# 6. Detect duplicates
# --------------------------------------------------

paired_examples = list(zip(descriptions, regex_targets))

unique_pairs = set(paired_examples)

description_counts = Counter(descriptions)
target_counts = Counter(regex_targets)

duplicate_pairs = total_examples - len(unique_pairs)

duplicate_descriptions = sum(
    count - 1 for count in description_counts.values() if count > 1
)

duplicate_targets = sum(count - 1 for count in target_counts.values() if count > 1)

print("\n========== DUPLICATE ANALYSIS ==========")

print("Unique input-target pairs:", len(unique_pairs))
print("Repeated input-target pairs:", duplicate_pairs)

print("Repeated descriptions:", duplicate_descriptions)
print("Repeated regex targets:", duplicate_targets)


# --------------------------------------------------
# 7. Find descriptions with conflicting targets
# --------------------------------------------------

description_to_targets = defaultdict(set)

for description, target in paired_examples:
    description_to_targets[description].add(target)

conflicting_descriptions = {
    description: targets
    for description, targets in description_to_targets.items()
    if len(targets) > 1
}

print("Descriptions with multiple distinct targets:", len(conflicting_descriptions))


# --------------------------------------------------
# 8. Character-length statistics
# --------------------------------------------------

source_lengths = [len(item) for item in descriptions]
target_lengths = [len(item) for item in regex_targets]


def print_length_statistics(name, values):

    print(f"\n{name}")

    print("Minimum:", min(values))
    print("Maximum:", max(values))
    print("Mean:", round(statistics.mean(values), 2))
    print("Median:", statistics.median(values))

    print("P90:", round(percentile(values, 90), 2))
    print("P95:", round(percentile(values, 95), 2))
    print("P99:", round(percentile(values, 99), 2))


print("\n========== LENGTH ANALYSIS ==========")

print_length_statistics(
    "Description character lengths",
    source_lengths,
)

print_length_statistics(
    "Regex character lengths",
    target_lengths,
)


# --------------------------------------------------
# 9. Inspect regex special characters
# --------------------------------------------------

special_characters = Counter(
    character
    for regex in regex_targets
    for character in regex
    if not character.isalnum() and not character.isspace()
)

print("\n========== REGEX CHARACTER FREQUENCY ==========")

for character, count in special_characters.most_common(25):
    print(repr(character), ":", count)


# --------------------------------------------------
# 10. Display longest examples
# --------------------------------------------------

print("\n========== LONGEST EXAMPLES ==========")

longest_indices = sorted(
    range(total_examples),
    key=lambda i: len(descriptions[i]) + len(regex_targets[i]),
    reverse=True,
)[:5]

for index in longest_indices:

    print(f"\nExample {index + 1}")

    print("Description:", descriptions[index])
    print("Regex:", regex_targets[index])

    print("Description characters:", len(descriptions[index]))
    print("Regex characters:", len(regex_targets[index]))


# --------------------------------------------------
# 11. Final summary
# --------------------------------------------------

print("\n========== AUDIT SUMMARY ==========")

print("Total examples:", total_examples)

print("Empty records:", len(empty_descriptions) + len(empty_targets))

print("Repeated pairs:", duplicate_pairs)

print("Conflicting descriptions:", len(conflicting_descriptions))

print("\nDataset audit completed.")
