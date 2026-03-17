"""
Token-based cost estimation per model.

Queries OpenRouter API for live pricing and computes estimates
based on actual prompt lengths from the dataset.

Usage:
    python -m analysis.estimate_cost [--grader MODEL]
"""

import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import yaml

try:
    import requests
except ImportError:
    requests = None


CONFIGS_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"
DATA_PATH = Path(__file__).parent.parent / "data" / "dataset.csv"

# Fallback pricing (per token) if API unavailable
FALLBACK_PRICING = {
    "openrouter/meta-llama/llama-3.1-8b-instruct": {"input": 0.06e-6, "output": 0.06e-6},
    "openrouter/google/gemma-2-9b-it": {"input": 0.08e-6, "output": 0.08e-6},
    "openrouter/qwen/qwen-2.5-7b-instruct": {"input": 0.06e-6, "output": 0.06e-6},
    "openrouter/mistralai/mistral-7b-instruct": {"input": 0.06e-6, "output": 0.06e-6},
    "openrouter/microsoft/phi-3-mini-128k-instruct": {"input": 0.10e-6, "output": 0.10e-6},
    "openrouter/google/gemini-2.0-flash-001": {"input": 0.075e-6, "output": 0.30e-6},
    "anthropic/claude-haiku-4-5-20241022": {"input": 0.80e-6, "output": 4.0e-6},
}

# Rough tokens-per-char ratio
CHARS_PER_TOKEN = 4


def load_config():
    with open(CONFIGS_PATH) as f:
        return yaml.safe_load(f)


def load_dataset():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch_openrouter_pricing() -> dict:
    """Fetch live pricing from OpenRouter API."""
    if requests is None:
        print("  (requests not installed, using fallback pricing)")
        return {}

    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        pricing = {}
        for m in models:
            model_id = f"openrouter/{m['id']}"
            p = m.get("pricing", {})
            if p.get("prompt") and p.get("completion"):
                pricing[model_id] = {
                    "input": float(p["prompt"]),
                    "output": float(p["completion"]),
                }
        return pricing
    except Exception as e:
        print(f"  (Could not fetch OpenRouter pricing: {e}, using fallback)")
        return {}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def main():
    parser = argparse.ArgumentParser(description="Estimate eval costs")
    parser.add_argument("--grader", type=str, default=None, help="Override grader model")
    args = parser.parse_args()

    config = load_config()
    grader_model = args.grader or config["grader_model"]
    eval_models = config["eval_models"]

    rows = load_dataset()

    # Filter to splittable samples
    if "chunks" in rows[0]:
        splittable = [r for r in rows if r.get("chunks") and r.get("num_chunks") and int(r["num_chunks"]) > 0]
    else:
        splittable = rows  # Before splitting, estimate based on all

    n_samples = len(splittable)
    print(f"Dataset: {len(rows)} total, {n_samples} splittable samples\n")

    # Calculate average prompt lengths
    avg_prompt_tokens = sum(estimate_tokens(r["adversarial_prompt"]) for r in splittable) / max(n_samples, 1)
    total_prompt_tokens = sum(estimate_tokens(r["adversarial_prompt"]) for r in splittable)

    # Estimate split tokens (multi-turn has more overhead)
    if "chunks" in rows[0]:
        total_split_tokens = 0
        for r in splittable:
            chunks = json.loads(r["chunks"])
            # Each turn: growing context + new chunk
            running = 0
            for chunk in chunks:
                running += estimate_tokens(chunk)
                total_split_tokens += running + 100  # 100 tokens for response per turn
    else:
        total_split_tokens = total_prompt_tokens * 3  # Rough estimate

    avg_output_tokens = 150  # Estimated average response length

    # Fetch pricing
    print("Fetching OpenRouter pricing...")
    live_pricing = fetch_openrouter_pricing()
    pricing = {**FALLBACK_PRICING, **live_pricing}

    # Eval model costs
    print("=" * 70)
    print("EVAL MODEL COSTS (per rollout)")
    print("=" * 70)

    total_eval_cost = 0
    for model_cfg in eval_models:
        model_id = model_cfg["id"]
        model_name = model_cfg["name"]
        p = pricing.get(model_id, {"input": 0.1e-6, "output": 0.1e-6})

        # Single-message cost
        single_input_cost = total_prompt_tokens * p["input"]
        single_output_cost = n_samples * avg_output_tokens * p["output"]
        single_cost = single_input_cost + single_output_cost

        # Split-message cost
        split_input_cost = total_split_tokens * p["input"]
        split_output_cost = total_split_tokens * 0.3 * p["output"]  # Responses at each turn
        split_cost = split_input_cost + split_output_cost

        model_total = single_cost + split_cost
        total_eval_cost += model_total

        print(f"\n{model_name} ({model_id})")
        print(f"  Input price: ${p['input']*1e6:.3f}/M tokens, Output: ${p['output']*1e6:.3f}/M tokens")
        print(f"  Single-message: ${single_cost:.4f}")
        print(f"  Split-message:  ${split_cost:.4f}")
        print(f"  Subtotal:       ${model_total:.4f}")

    # Grader costs
    print("\n" + "=" * 70)
    print("GRADER COSTS")
    print("=" * 70)

    n_grading_calls = n_samples * 2 * len(eval_models)  # 2 conditions x N models
    grader_input_tokens = n_grading_calls * 800  # ~800 tokens per grading call
    grader_output_tokens = n_grading_calls * 100  # ~100 tokens per response

    for gm_id, gm_name in [
        ("openrouter/google/gemini-2.0-flash-001", "Gemini Flash"),
        ("anthropic/claude-haiku-4-5-20241022", "Haiku 4.5"),
    ]:
        p = pricing.get(gm_id, {"input": 0.1e-6, "output": 0.1e-6})
        cost = grader_input_tokens * p["input"] + grader_output_tokens * p["output"]
        marker = " <-- selected" if gm_id == grader_model else ""
        print(f"\n{gm_name} ({gm_id}){marker}")
        print(f"  {n_grading_calls} grading calls")
        print(f"  Input: {grader_input_tokens:,} tokens, Output: {grader_output_tokens:,} tokens")
        print(f"  Cost: ${cost:.2f}")

    # Selected grader cost
    gp = pricing.get(grader_model, {"input": 0.1e-6, "output": 0.1e-6})
    grader_cost = grader_input_tokens * gp["input"] + grader_output_tokens * gp["output"]

    # Total
    total = total_eval_cost + grader_cost
    print("\n" + "=" * 70)
    print(f"TOTAL ESTIMATED COST: ${total:.2f}")
    print(f"  Eval models: ${total_eval_cost:.2f}")
    print(f"  Grader ({grader_model}): ${grader_cost:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
