#!/usr/bin/env python3
"""Build a larger prompt bench for empirical routing experiments.

The original paper uses 60 hand-written prompts.  This script expands that
seed set into a deterministic benchmark with domain, complexity, length, and
expected-output controls.  It writes outside Dropbox by default; set
EDGELLM_EXPERIMENT_DIR to override.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


DEFAULT_OUT = Path.home() / "codex-experiments" / "edgellm-router" / "results"


@dataclass(frozen=True)
class PromptSeed:
    domain: str
    complexity: int
    category: str
    text: str
    expected_keywords: str


BASE_SEEDS: List[PromptSeed] = [
    PromptSeed("narrative", 0, "short_dialogue", "The player greets the shopkeeper.", "greet,welcome"),
    PromptSeed("narrative", 0, "location", "Where is the blacksmith's shop?", "direction,shop"),
    PromptSeed("narrative", 1, "lore", "Tell me about the history of the Elven Kingdom.", "history,elves"),
    PromptSeed("narrative", 1, "quest", "The guild master explains the next mission objectives.", "mission,objective"),
    PromptSeed("narrative", 2, "battle", "Narrate an intense battle with a dragon using tactics and terrain.", "dragon,tactic"),
    PromptSeed("narrative", 2, "council", "Narrate a council meeting where five factions debate the war.", "council,faction"),
    PromptSeed("customer_qa", 0, "hours", "What are your opening hours?", "hours"),
    PromptSeed("customer_qa", 0, "shipping", "Do you offer free shipping?", "shipping"),
    PromptSeed("customer_qa", 1, "refund", "I need a refund because my item arrived damaged.", "refund,damaged"),
    PromptSeed("customer_qa", 1, "billing", "Why was I charged twice this month?", "billing,charge"),
    PromptSeed("customer_qa", 2, "compliance", "Explain GDPR obligations when exporting customer data abroad.", "gdpr,data"),
    PromptSeed("customer_qa", 2, "incident", "Write a post-mortem for an outage that affected 20 percent of users.", "post-mortem,outage"),
    PromptSeed("code", 0, "snippet", "Print 'Hello, world!' in Python.", "print,hello"),
    PromptSeed("code", 0, "rename", "Rename variable foo to bar in this function.", "rename,bar"),
    PromptSeed("code", 1, "test", "Write a pytest for the sort_and_dedupe function.", "pytest,test"),
    PromptSeed("code", 1, "async", "Convert this synchronous HTTP call into an asyncio coroutine.", "asyncio,http"),
    PromptSeed("code", 2, "rate_limit", "Implement a deterministic rate limiter with sliding window and token bucket modes.", "rate,token"),
    PromptSeed("code", 2, "migration", "Write a safe database migration that backfills a NOT NULL column on a 50M row table.", "migration,backfill"),
    PromptSeed("document_qa", 0, "lookup", "In one sentence, state whether the policy allows refunds.", "refund,policy"),
    PromptSeed("document_qa", 1, "summarise", "Summarise the attached policy section and list the two operational risks.", "summarise,risk"),
    PromptSeed("document_qa", 2, "synthesis", "Compare three policy sections, identify contradictions, and draft a corrected version.", "compare,contradiction"),
]


LENGTH_PACKS = {
    "short": "",
    "medium": " Use two to four sentences and include one concrete example.",
    "long": (
        " Provide a structured answer with context, assumptions, a concise plan, "
        "and a final recommendation. Keep it practical and avoid generic filler."
    ),
}


CONTEXT_SNIPPETS = [
    "Previous turn: the user asked for a fast answer and is waiting in an interactive session.",
    "System state: network latency is variable and the application prefers answers under one second.",
    "Background: the response should be helpful but not verbose unless the task is complex.",
    "Constraint: do not invent external facts; state assumptions clearly when needed.",
]


def default_output_dir() -> Path:
    return Path(os.environ.get("EDGELLM_EXPERIMENT_DIR", DEFAULT_OUT))


def make_rows(target_size: int) -> Iterable[dict]:
    row_id = 0
    repeats = max(1, (target_size + len(BASE_SEEDS) - 1) // len(BASE_SEEDS))
    for repeat in range(repeats):
        for seed in BASE_SEEDS:
            length_name = ["short", "medium", "long"][(repeat + seed.complexity) % 3]
            ctx = CONTEXT_SNIPPETS[(repeat + len(seed.category)) % len(CONTEXT_SNIPPETS)]
            variant = repeat % 11
            prompt = (
                f"{ctx}\n"
                f"Task variant {variant}: {seed.text}{LENGTH_PACKS[length_name]}"
            )
            row_id += 1
            yield {
                "id": row_id,
                "domain": seed.domain,
                "complexity": seed.complexity,
                "category": seed.category,
                "length_bucket": length_name,
                "expected_output_tokens": {"short": 48, "medium": 96, "long": 192}[length_name],
                "expected_keywords": seed.expected_keywords,
                "prompt": prompt,
            }
            if row_id >= target_size:
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--filename", default="extended_prompt_bench.csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.filename
    rows = list(make_rows(args.size))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} prompts to {out_path}")


if __name__ == "__main__":
    main()
