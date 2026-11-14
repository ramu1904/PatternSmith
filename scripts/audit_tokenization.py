from pathlib import Path
import json

import numpy as np
import tiktoken

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "NL-RX-Turk"

SOURCE_FILE = DATA_DIR / "src.txt"
TARGET_FILE = DATA_DIR / "targ.txt"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

descriptions = SOURCE_FILE.read_text(encoding="utf-8").splitlines()

targets = TARGET_FILE.read_text(encoding="utf-8").splitlines()

if len(descriptions) != len(targets):
    raise ValueError("Description and target counts do not match.")

if not descriptions:
    raise ValueError("Dataset is empty.")


# --------------------------------------------------
# 3. Load GPT-2 tokenizer
# --------------------------------------------------

encoding = tiktoken.get_encoding("gpt2")

EOT_TOKEN_ID = encoding.eot_token
VOCAB_SIZE = encoding.n_vocab


# --------------------------------------------------
# 4. Tokenize each example
# --------------------------------------------------

prompt_lengths = []
target_lengths = []
total_lengths = []

tokenization_examples = []

for description, target in zip(descriptions, targets):

    prompt = f"Description: {description}\nRegex:"

    prompt_ids = encoding.encode(prompt)
    target_ids = encoding.encode(target)

    total_ids = prompt_ids + target_ids + [EOT_TOKEN_ID]

    prompt_lengths.append(len(prompt_ids))
    target_lengths.append(len(target_ids))
    total_lengths.append(len(total_ids))

    tokenization_examples.append(
        {
            "description": description,
            "target": target,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
            "total_ids": total_ids,
        }
    )


# --------------------------------------------------
# 5. Calculate sequence statistics
# --------------------------------------------------


def calculate_statistics(values):

    values = np.array(values)

    return {
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": round(float(np.mean(values)), 2),
        "median": round(float(np.median(values)), 2),
        "p90": int(np.percentile(values, 90)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
    }


prompt_stats = calculate_statistics(prompt_lengths)
target_stats = calculate_statistics(target_lengths)
total_stats = calculate_statistics(total_lengths)


# --------------------------------------------------
# 6. Check candidate context lengths
# --------------------------------------------------

candidate_context_lengths = [64, 128, 256, 512]

context_results = {}

for context_length in candidate_context_lengths:

    fitting_examples = sum(length <= context_length for length in total_lengths)

    context_results[str(context_length)] = {
        "fitting_examples": fitting_examples,
        "truncated_examples": len(total_lengths) - fitting_examples,
        "coverage_percentage": round(
            fitting_examples / len(total_lengths) * 100,
            2,
        ),
    }


# --------------------------------------------------
# 7. Inspect example tokenization
# --------------------------------------------------

print("\n========== TOKENIZER INFORMATION ==========")

print("Tokenizer: GPT-2 BPE")
print("Vocabulary size:", VOCAB_SIZE)
print("EOT token ID:", EOT_TOKEN_ID)

print("\n========== SAMPLE TOKENIZATION ==========")

for index in [0, 1, 2]:

    example = tokenization_examples[index]

    print(f"\nExample {index + 1}")

    print("Description:", example["description"])
    print("Target:", example["target"])

    print("Prompt token count:", len(example["prompt_ids"]))
    print("Target token count:", len(example["target_ids"]))

    print("Prompt tokens:")
    print(
        encoding.decode_single_token_bytes(example["prompt_ids"][0])
        if len(example["prompt_ids"]) == 1
        else [
            encoding.decode_single_token_bytes(token).decode(
                "utf-8",
                errors="replace",
            )
            for token in example["prompt_ids"]
        ]
    )

    print("Prompt IDs:", example["prompt_ids"])
    print("Target IDs:", example["target_ids"])

    print("Total sequence length:", len(example["total_ids"]))


# --------------------------------------------------
# 8. Print statistics
# --------------------------------------------------

print("\n========== TOKEN LENGTH STATISTICS ==========")

print("\nPrompt lengths:")
print(json.dumps(prompt_stats, indent=4))

print("\nTarget lengths:")
print(json.dumps(target_stats, indent=4))

print("\nTotal sequence lengths:")
print(json.dumps(total_stats, indent=4))


# --------------------------------------------------
# 9. Context length coverage
# --------------------------------------------------

print("\n========== CONTEXT LENGTH COVERAGE ==========")

for context_length, result in context_results.items():

    print(f"\nContext length: {context_length}")

    for key, value in result.items():
        print(f"{key}: {value}")


# --------------------------------------------------
# 10. Save audit report
# --------------------------------------------------

report = {
    "tokenizer": "GPT-2 BPE",
    "vocabulary_size": VOCAB_SIZE,
    "eot_token_id": EOT_TOKEN_ID,
    "dataset_examples": len(descriptions),
    "prompt_statistics": prompt_stats,
    "target_statistics": target_stats,
    "total_sequence_statistics": total_stats,
    "context_length_coverage": context_results,
}

report_path = DATA_DIR / "tokenization_audit.json"

report_path.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8",
)

print("\nAudit report saved to:", report_path)
