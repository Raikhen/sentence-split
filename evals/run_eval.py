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
    parser.add_argument("--rollouts", type=int, default=1,
                        help="Number of rollouts per model/condition")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (use >0 for multiple rollouts)")
    parser.add_argument("--min-split-quality", type=int, default=None,
                        help="Minimum split_quality score to include (e.g. 5)")
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

    temperature = args.temperature
    if args.rollouts > 1 and temperature == 0.0:
        temperature = 1.0
        print(f"Note: temperature auto-set to {temperature} for multiple rollouts")

    model_ids = [m["id"] for m in eval_models]
    model_names = [m["name"] for m in eval_models]

    print(f"Running evals with {len(eval_models)} models x {len(conditions)} conditions x {args.rollouts} epochs")
    print(f"Models: {', '.join(model_names)}")
    print(f"Grader model: {grader_model}")
    print(f"Temperature: {temperature}")
    if args.min_split_quality is not None:
        print(f"Min split quality: {args.min_split_quality}")
    if args.limit:
        print(f"Sample limit: {args.limit}")
    print()

    task_kwargs = {"min_split_quality": args.min_split_quality}

    for model_cfg in eval_models:
        model_id = model_cfg["id"]
        model_name = model_cfg["name"]

        tasks = []
        if "single" in conditions:
            tasks.append(single_message_eval(grader_model=grader_model, **task_kwargs))
        if "split" in conditions:
            tasks.append(split_message_eval(grader_model=grader_model, **task_kwargs))

        print(f"Running {model_name} ({len(tasks)} tasks x {args.rollouts} epochs)...")
        eval(
            tasks,
            model=model_id,
            temperature=temperature,
            log_dir=str(LOG_DIR),
            limit=args.limit,
            epochs=args.rollouts,
            max_samples=50,
            max_tasks=1,
            display="plain",
            fail_on_error=0.25,
            retry_on_error=3,
        )
        print(f"Completed {model_name}\n")

    print("\nAll evals complete. Logs saved to:", LOG_DIR)


if __name__ == "__main__":
    main()
