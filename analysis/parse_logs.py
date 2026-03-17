"""
Utility to extract results from Inspect AI eval logs into a DataFrame.

Usage:
    python -m analysis.parse_logs [--log-dir ./logs]
"""

import argparse
import json
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


LOG_DIR = Path(__file__).parent.parent / "logs"


def parse_eval_log(log_path: Path) -> list[dict]:
    """Parse a single Inspect AI eval log file."""
    with open(log_path, "r") as f:
        log_data = json.load(f)

    results = []
    # Extract model and task info
    eval_info = log_data.get("eval", {})
    model = eval_info.get("model", "unknown")
    task_name = eval_info.get("task", "unknown")

    # Determine condition from task name
    if "single" in task_name.lower():
        condition = "single"
    elif "split" in task_name.lower():
        condition = "split"
    else:
        condition = "unknown"

    # Extract per-sample results
    samples = log_data.get("samples", [])
    for sample in samples:
        sample_id = sample.get("id", "")
        metadata = sample.get("metadata", {})

        # Get score
        scores = sample.get("scores", {})
        score_data = None
        for scorer_name, scorer_result in scores.items():
            score_data = scorer_result
            break

        if score_data is None:
            continue

        value = score_data.get("value", "")
        answer = score_data.get("answer", "")
        score_metadata = score_data.get("metadata", {})

        refused = score_metadata.get("refused", None)
        if refused is None:
            refused = value == "C"

        results.append({
            "model": model,
            "condition": condition,
            "sample_id": sample_id,
            "refused": refused,
            "score_value": value,
            "score_answer": answer,
            "risk_domain": metadata.get("risk_domain", ""),
            "risk_subdomain": metadata.get("risk_subdomain", ""),
            "num_chunks": metadata.get("num_chunks", 0),
            "split_quality": metadata.get("split_quality", ""),
            "triggered_criteria": score_metadata.get("triggered_criteria", []),
        })

    return results


def parse_all_logs(log_dir: Path) -> list[dict]:
    """Parse all eval log files in a directory."""
    all_results = []
    log_files = sorted(log_dir.glob("*.json"))

    if not log_files:
        # Also check for .eval files (Inspect AI format)
        log_files = sorted(log_dir.glob("*.eval"))

    print(f"Found {len(log_files)} log files in {log_dir}")

    for log_path in log_files:
        try:
            results = parse_eval_log(log_path)
            all_results.extend(results)
            print(f"  {log_path.name}: {len(results)} samples")
        except Exception as e:
            print(f"  {log_path.name}: ERROR - {e}")

    return all_results


def results_to_dataframe(results: list[dict]):
    """Convert results to a pandas DataFrame."""
    if pd is None:
        raise ImportError("pandas is required for DataFrame output. Install with: pip install pandas")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Parse Inspect AI eval logs")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR), help="Directory containing eval logs")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return

    results = parse_all_logs(log_dir)

    if not results:
        print("No results found.")
        return

    print(f"\nTotal results: {len(results)}")

    if pd is not None:
        df = results_to_dataframe(results)
        print("\nSummary:")
        print(df.groupby(["model", "condition"])["refused"].mean().unstack())

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved to {args.output}")
    else:
        # Print without pandas
        print("\nResults (install pandas for better output):")
        for r in results[:10]:
            print(f"  {r['model']} | {r['condition']} | {r['sample_id']} | refused={r['refused']}")
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more")


if __name__ == "__main__":
    main()
