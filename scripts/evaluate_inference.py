import argparse
import csv
import random
from pathlib import Path

from patternsmith.inference import load_patternsmith


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    test_dir = project_root / "data" / "splits" / "NL-RX-Turk" / "test"

    source_path = test_dir / "src.txt"
    target_path = test_dir / "targ.txt"

    descriptions = source_path.read_text(encoding="utf-8").splitlines()

    references = target_path.read_text(encoding="utf-8").splitlines()

    if len(descriptions) != len(references):
        raise ValueError("Source and target counts do not match.")

    if not descriptions:
        raise ValueError("Test dataset is empty.")

    # Select a reproducible subset.
    indices = list(range(len(descriptions)))
    random.Random(args.seed).shuffle(indices)

    if args.limit > 0:
        indices = indices[: min(args.limit, len(indices))]

    # Load trained model.
    model = load_patternsmith()

    results = []
    exact_matches = 0
    empty_predictions = 0

    print(f"\nEvaluating {len(indices)} test examples...")
    print(f"Checkpoint epoch: {model.epoch}")
    print(f"Device: {model.device}\n")

    for position, index in enumerate(indices, start=1):

        description = descriptions[index]
        reference = references[index].strip()

        prediction = model.generate(
            description,
            max_new_tokens=64,
        )

        is_exact_match = prediction.strip() == reference

        exact_matches += int(is_exact_match)
        empty_predictions += int(not prediction.strip())

        results.append(
            {
                "index": index,
                "description": description,
                "reference_regex": reference,
                "predicted_regex": prediction,
                "exact_match": is_exact_match,
            }
        )

        if position % 10 == 0 or position == len(indices):
            print(f"Processed {position}/{len(indices)} examples")

    total = len(results)
    accuracy = exact_matches / total

    # Save detailed predictions.
    output_dir = project_root / "outputs" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "test_predictions.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "description",
                "reference_regex",
                "predicted_regex",
                "exact_match",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("PATTERNSMITH TEST EVALUATION")
    print("=" * 60)

    print(f"Total evaluated       : {total}")
    print(f"Exact matches         : {exact_matches}")
    print(f"Incorrect predictions : {total - exact_matches}")
    print(f"Empty predictions     : {empty_predictions}")
    print(f"Exact-match accuracy  : {accuracy:.2%}")

    print(f"\nResults saved to:\n{output_path}")

    print("\nFIRST 15 PREDICTIONS")
    print("-" * 60)

    for result in results[:15]:
        print(f"\nDescription : {result['description']}")
        print(f"Reference   : {result['reference_regex']}")
        print(f"Prediction  : {result['predicted_regex']}")
        print(f"Exact match : {result['exact_match']}")


if __name__ == "__main__":
    main()
