"""
Inspect AI task definitions for single-message and split-message evaluations.

Both tasks filter to only splittable samples so the comparison is paired
on identical prompts.
"""

import csv
import json
from pathlib import Path

from inspect_ai import task, Task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate

from evals.scorers import refusal_scorer
from evals.solvers import split_message_solver


DATA_PATH = Path(__file__).parent.parent / "data" / "dataset.csv"


def load_splittable_samples() -> list[dict]:
    """Load only samples that were successfully split."""
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Only include rows with valid chunks
            if row.get("chunks") and row.get("num_chunks") and int(row["num_chunks"]) > 0:
                rows.append(row)
    return rows


def make_single_message_samples(rows: list[dict]) -> list[Sample]:
    """Create samples for single-message evaluation."""
    samples = []
    for row in rows:
        samples.append(
            Sample(
                id=row["ID"],
                input=row["adversarial_prompt"],
                metadata={
                    "adversarial_prompt": row["adversarial_prompt"],
                    "rubric": row["rubric"],
                    "risk_domain": row["risk_domain"],
                    "risk_subdomain": row["risk_subdomain"],
                    "num_chunks": int(row["num_chunks"]),
                    "split_quality": row.get("split_quality", ""),
                    "condition": "single",
                },
            )
        )
    return samples


def make_split_message_samples(rows: list[dict]) -> list[Sample]:
    """Create samples for split-message evaluation.

    The first chunk is used as the initial input message.
    All chunks are stored in metadata for the solver.
    """
    samples = []
    for row in rows:
        chunks = json.loads(row["chunks"])
        if not chunks:
            continue
        samples.append(
            Sample(
                id=row["ID"],
                input=chunks[0],  # First chunk as initial message
                metadata={
                    "adversarial_prompt": row["adversarial_prompt"],
                    "chunks": row["chunks"],
                    "rubric": row["rubric"],
                    "risk_domain": row["risk_domain"],
                    "risk_subdomain": row["risk_subdomain"],
                    "num_chunks": int(row["num_chunks"]),
                    "split_quality": row.get("split_quality", ""),
                    "condition": "split",
                },
            )
        )
    return samples


@task
def single_message_eval(grader_model: str = "openrouter/google/gemini-2.0-flash-001"):
    """Evaluate refusal rates when adversarial prompts are sent as a single message."""
    rows = load_splittable_samples()
    samples = make_single_message_samples(rows)
    return Task(
        dataset=MemoryDataset(samples=samples, name="fortress-single"),
        solver=generate(),
        scorer=refusal_scorer(grader_model=grader_model),
    )


@task
def split_message_eval(grader_model: str = "openrouter/google/gemini-2.0-flash-001"):
    """Evaluate refusal rates when adversarial prompts are split across multiple turns."""
    rows = load_splittable_samples()
    samples = make_split_message_samples(rows)
    return Task(
        dataset=MemoryDataset(samples=samples, name="fortress-split"),
        solver=split_message_solver(),
        scorer=refusal_scorer(grader_model=grader_model),
    )
