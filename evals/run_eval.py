"""
Orchestration script: runs all (model x condition) eval combos.

Usage:
    python -m evals.run_eval [--limit N] [--models model1,model2] [--grader MODEL]
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
import yaml
from inspect_ai import eval

load_dotenv(Path(__file__).parent.parent / ".env")

from evals.task import single_message_eval, split_message_eval


CONFIGS_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"
LOG_DIR = Path(__file__).parent.parent / "logs"


def load_config():
    with open(CONFIGS_PATH) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run sentence-split evals")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of model names to run (filters config)")
    parser.add_argument("--grader", type=str, default=None,
                        help="Override grader model ID")
    parser.add_argument("--conditions", type=str, default="single,split",
                        help="Comma-separated conditions to run (single, split, or both)")
    args = parser.parse_args()

    config = load_config()
    grader_model = args.grader or config["grader_model"]

    # Filter models if specified
    eval_models = config["eval_models"]
    if args.models:
        model_filter = set(args.models.split(","))
        eval_models = [m for m in eval_models if m["name"] in model_filter or m["id"] in model_filter]

    conditions = set(args.conditions.split(","))

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build task kwargs
    task_kwargs = {}
    if args.limit:
        task_kwargs["limit"] = args.limit

    print(f"Running evals with {len(eval_models)} models x {len(conditions)} conditions")
    print(f"Grader model: {grader_model}")
    if args.limit:
        print(f"Sample limit: {args.limit}")
    print()

    for model_cfg in eval_models:
        model_id = model_cfg["id"]
        model_name = model_cfg["name"]

        if "single" in conditions:
            print(f"[Single] {model_name} ({model_id})")
            task = single_message_eval(grader_model=grader_model)
            eval(
                task,
                model=model_id,
                temperature=0.0,
                log_dir=str(LOG_DIR),
                limit=args.limit,
            )

        if "split" in conditions:
            print(f"[Split]  {model_name} ({model_id})")
            task = split_message_eval(grader_model=grader_model)
            eval(
                task,
                model=model_id,
                temperature=0.0,
                log_dir=str(LOG_DIR),
                limit=args.limit,
            )

    print("\nAll evals complete. Logs saved to:", LOG_DIR)


if __name__ == "__main__":
    main()
