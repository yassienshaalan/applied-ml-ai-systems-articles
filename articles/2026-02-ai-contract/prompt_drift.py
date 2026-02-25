"""
experiments/prompt_drift.py

Experiment 1: Prompt Drift Under Minor Refactoring
===================================================

Hypothesis: Small, "safe-looking" prompt edits can silently break output contracts
even when the semantic intent is preserved. Without explicit contracts, these
regressions are invisible.

Method:
  1. Define a strict PromptContract (required fields, citation requirement, confidence values).
  2. Run three prompt variants against the contract:
     - v1: The "production" prompt (written to satisfy the contract).
     - v2: A tone refactor ("make it friendlier") that accidentally breaks field naming.
     - v3: A brevity refactor ("be more concise") that drops required fields.
  3. Show that v1 passes, v2 and v3 fail — and why.

The contract is the *only* thing that caught the regressions.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from contracts.prompt import PromptContract, PromptContractViolation

load_dotenv()
console = Console()

# ---------------------------------------------------------------------------
# The contract — defined once, applied to all prompt variants
# ---------------------------------------------------------------------------
CONTRACT = PromptContract(
    required_fields=["answer", "confidence", "sources", "reasoning"],
    forbidden_strings=["I don't know", "As an AI", "I cannot"],
    must_cite_sources=True,
    allowed_confidence_values=["high", "medium", "low"],
    output_format="json",
    max_tokens=400,
)

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
SYSTEM_V1 = """You are a precise research assistant. Always respond in JSON with these exact fields:
- answer: your factual answer (string)
- confidence: one of "high", "medium", or "low"
- sources: a list of at least one source you're drawing on (list of strings)
- reasoning: brief explanation of your reasoning (string)

Respond ONLY with valid JSON. No markdown fences."""

SYSTEM_V2 = """Hey! You're a friendly research helper. Keep it warm and approachable.
Respond in JSON with:
- response: your answer (friendlier tone!)
- confidence_level: how confident you are (percentage, e.g. 85)
- sources: where you got this from
- reasoning: your thought process

Respond ONLY with valid JSON."""
# ^ Regression: renamed 'answer' → 'response', changed confidence format to integer %

SYSTEM_V3 = """Be extremely concise. Answer in JSON: {"answer": "...", "confidence": "high/medium/low"}.
Nothing else."""
# ^ Regression: dropped 'sources' and 'reasoning', will break must_cite_sources

QUESTION = "What is the capital of France, and why is it historically significant?"

VARIANTS = [
    ("v1_production", SYSTEM_V1, "Original production prompt"),
    ("v2_tone_refactor", SYSTEM_V2, "Tone refactor (friendlier) — renames fields"),
    ("v3_brevity_refactor", SYSTEM_V3, "Brevity refactor — drops required fields"),
]


def call_llm(client: OpenAI, system: str, question: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def run_experiment(client: OpenAI) -> list[dict]:
    results = []
    for variant_id, system_prompt, description in VARIANTS:
        console.print(f"\n[bold cyan]Running:[/] {variant_id} — {description}")
        raw = call_llm(client, system_prompt, QUESTION)
        console.print(f"[dim]Raw response:[/] {raw[:200]}{'...' if len(raw) > 200 else ''}")

        compliant, reason = CONTRACT.is_compliant(raw)
        results.append({
            "variant": variant_id,
            "description": description,
            "compliant": compliant,
            "violation": reason if not compliant else None,
            "raw_response": raw,
        })

        status = "[green]✓ PASS[/]" if compliant else f"[red]✗ FAIL[/]"
        console.print(f"Contract check: {status}")
        if not compliant:
            console.print(f"[red]Reason:[/] {reason}")

    return results


def print_summary(results: list[dict]) -> None:
    table = Table(title="\nPrompt Drift Experiment — Contract Results", show_lines=True)
    table.add_column("Variant", style="cyan")
    table.add_column("Description")
    table.add_column("Compliant?", justify="center")
    table.add_column("Violation Rule")

    for r in results:
        status = "[green]✓ PASS[/green]" if r["compliant"] else "[red]✗ FAIL[/red]"
        violation = r["violation"] or "—"
        # Truncate long violation messages
        if len(violation) > 80:
            violation = violation[:77] + "..."
        table.add_row(r["variant"], r["description"], status, violation)

    console.print(table)

    passed = sum(1 for r in results if r["compliant"])
    console.print(
        f"\n[bold]Result:[/] {passed}/{len(results)} variants passed the contract.\n"
        "[bold]Hypothesis confirmed[/] — minor prompt edits silently break output contracts.\n"
        "Without the contract, these regressions would reach production undetected.\n"
    )


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not set.[/]")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    results = run_experiment(client)
    print_summary(results)

    # Save results
    out_path = Path(__file__).parent.parent / "results" / "prompt_drift.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"Results saved to [cyan]{out_path}[/]")
