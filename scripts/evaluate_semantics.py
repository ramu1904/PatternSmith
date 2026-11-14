import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from regexbench import Dialect, Semantics, equivalent


def evaluate_one(item):
    index, row = item
    reference = row["reference_regex"].strip()
    prediction = row["predicted_regex"].strip()

    if not prediction:
        verdict = "EMPTY"
        witness = ""
    else:
        try:
            result = equivalent(
                prediction,
                reference,
                dialect=Dialect.BRICS,
                semantics=Semantics.FULLMATCH,
            )
            verdict = result.verdict.name
            witness = result.witness or ""
        except Exception as exc:
            verdict = "ERROR"
            witness = str(exc)

    row["semantic_verdict"] = verdict
    row["distinguishing_witness"] = witness

    return index, row, verdict


def main():
    project_root = Path(__file__).resolve().parents[1]

    input_path = project_root / "outputs" / "inference" / "test_predictions.csv"
    output_path = project_root / "outputs" / "inference" / "semantic_evaluation.csv"

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        original_fields = reader.fieldnames

    if not rows:
        print("No predictions found.")
        return

    fieldnames = list(original_fields)
    for field in ("semantic_verdict", "distinguishing_witness"):
        if field not in fieldnames:
            fieldnames.append(field)

    counts = Counter()
    completed = 0
    total = len(rows)

    print(f"Evaluating {total} regex pairs with 4 workers...", flush=True)

    # Write each completed result immediately.
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        file.flush()

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(evaluate_one, (i, row)) for i, row in enumerate(rows)
            ]

            try:
                for future in as_completed(futures):
                    index, row, verdict = future.result()

                    writer.writerow(row)
                    file.flush()

                    counts[verdict] += 1
                    completed += 1

                    if completed % 25 == 0 or completed == total:
                        print(
                            f"Progress: {completed}/{total} "
                            f"({completed / total:.1%})",
                            flush=True,
                        )

            except KeyboardInterrupt:
                print(
                    "\nInterrupted. Completed results are saved.",
                    flush=True,
                )
                for future in futures:
                    future.cancel()
                raise

    equivalent_count = counts["EQUIVALENT"]
    different_count = counts["DIFFERENT"]
    decided = equivalent_count + different_count

    print("\n" + "=" * 60)
    print("PATTERNSMITH SEMANTIC EVALUATION")
    print("=" * 60)
    print(f"Total examples          : {total}")
    print(f"Completed comparisons   : {completed}")
    print(f"Semantically equivalent : {equivalent_count}")
    print(f"Semantically different  : {different_count}")
    print(f"Undecidable             : {counts['UNDECIDABLE']}")
    print(f"Unsupported             : {counts['UNSUPPORTED']}")
    print(f"Errors                  : {counts['ERROR']}")
    print(f"Empty predictions       : {counts['EMPTY']}")

    print(f"\nVerified equivalence rate: {equivalent_count / total:.2%}")

    if decided:
        print(f"Equivalence among decided: " f"{equivalent_count / decided:.2%}")

    print(f"\nResults saved to:\n{output_path}")


if __name__ == "__main__":
    main()
