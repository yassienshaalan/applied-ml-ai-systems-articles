"""
experiments/tool_violations.py

Experiment 4: Semantic Violations in Tool Interactions
=======================================================

Hypothesis: LLMs can produce structurally valid tool calls that violate semantic
invariants — calls that pass JSON schema validation but encode logically impossible
or dangerous operations. Without semantic contracts, these slip through silently.

Method:
  1. Define a ToolContract for a "book_flight" tool with semantic rules:
     - return_date must be >= depart_date
     - passengers must be 1–9
     - cabin_class must be economy/business/first
     - no_infants_alone: infant (age<2) cannot travel without adult
  2. Ask the LLM to generate tool calls for several scenarios, including tricky ones.
  3. Run each call through the contract and show what would have executed vs what
     should have been rejected.
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

from contracts.tool import ToolContract, ToolContractViolation, ParameterConstraint

load_dotenv()
console = Console()

# ---------------------------------------------------------------------------
# Define the tool contract
# ---------------------------------------------------------------------------
FLIGHT_CONTRACT = ToolContract(
    tool_name="book_flight",
    parameter_constraints=[
        ParameterConstraint(name="origin", required=True, allowed_types=["str"], min_length=3, max_length=3),
        ParameterConstraint(name="destination", required=True, allowed_types=["str"], min_length=3, max_length=3),
        ParameterConstraint(name="depart_date", required=True, allowed_types=["str"]),
        ParameterConstraint(name="return_date", required=False, allowed_types=["str"]),
        ParameterConstraint(name="passengers", required=True, allowed_types=["int"], min_value=1, max_value=9),
        ParameterConstraint(name="cabin_class", required=True, allowed_values=["economy", "business", "first"]),
        ParameterConstraint(name="infant_passengers", required=False, allowed_types=["int"], min_value=0, max_value=4),
    ],
    forbidden_arg_combinations=[],
)

# Add semantic rules
FLIGHT_CONTRACT.add_semantic_rule(
    "return_after_departure",
    "return_date must be on or after depart_date",
    lambda args: args.get("return_date", "9999-99-99") >= args["depart_date"],
)
FLIGHT_CONTRACT.add_semantic_rule(
    "no_infant_without_adult",
    "Infant passengers require at least one adult (total passengers > infant count)",
    lambda args: (
        args.get("infant_passengers", 0) == 0 or
        args.get("passengers", 0) > args.get("infant_passengers", 0)
    ),
)
FLIGHT_CONTRACT.add_semantic_rule(
    "origin_ne_destination",
    "Origin and destination must be different airports",
    lambda args: args.get("origin", "").upper() != args.get("destination", "").upper(),
)

# ---------------------------------------------------------------------------
# Test cases — mix of valid and semantically invalid calls
# ---------------------------------------------------------------------------
TEST_CALLS = [
    {
        "scenario": "Valid round-trip",
        "args": {"origin": "LHR", "destination": "JFK", "depart_date": "2025-06-01",
                 "return_date": "2025-06-15", "passengers": 2, "cabin_class": "economy"},
        "should_pass": True,
    },
    {
        "scenario": "Return before departure (impossible trip)",
        "args": {"origin": "LHR", "destination": "JFK", "depart_date": "2025-06-15",
                 "return_date": "2025-06-01", "passengers": 1, "cabin_class": "economy"},
        "should_pass": False,
    },
    {
        "scenario": "Too many passengers (exceeds max)",
        "args": {"origin": "CDG", "destination": "LAX", "depart_date": "2025-07-01",
                 "passengers": 12, "cabin_class": "business"},
        "should_pass": False,
    },
    {
        "scenario": "Invalid cabin class (structurally valid JSON, semantically invalid)",
        "args": {"origin": "SYD", "destination": "SIN", "depart_date": "2025-08-10",
                 "passengers": 1, "cabin_class": "premium_economy"},  # not in allowed values
        "should_pass": False,
    },
    {
        "scenario": "Infants without adults (3 infants, 3 total — all infants!)",
        "args": {"origin": "DXB", "destination": "BOM", "depart_date": "2025-09-01",
                 "passengers": 3, "infant_passengers": 3, "cabin_class": "economy"},
        "should_pass": False,
    },
    {
        "scenario": "Same origin and destination",
        "args": {"origin": "LHR", "destination": "LHR", "depart_date": "2025-05-01",
                 "passengers": 1, "cabin_class": "first"},
        "should_pass": False,
    },
    {
        "scenario": "Valid one-way with infant and adult",
        "args": {"origin": "ORD", "destination": "MIA", "depart_date": "2025-10-01",
                 "passengers": 2, "infant_passengers": 1, "cabin_class": "economy"},
        "should_pass": True,
    },
]


def run_experiment() -> list[dict]:
    """Run all test cases against the contract without needing a live LLM."""
    results = []

    console.print("\n[bold cyan]Tool Contract Validation — book_flight[/]")
    console.print("Testing calls that are structurally valid JSON but may violate semantic invariants.\n")

    for tc in TEST_CALLS:
        scenario = tc["scenario"]
        args = tc["args"]
        should_pass = tc["should_pass"]

        try:
            FLIGHT_CONTRACT.validate(args)
            passed = True
            violation = None
        except ToolContractViolation as e:
            passed = False
            violation = str(e)

        correct = passed == should_pass
        status_icon = "[green]✓[/]" if passed else "[red]✗[/]"
        expectation = "[green]✓ CORRECT[/]" if correct else "[bold red]⚠ UNEXPECTED[/]"

        console.print(f"{status_icon} [bold]{scenario}[/]")
        if violation:
            console.print(f"   [red]Violation:[/] {violation}")
        console.print(f"   Expected: {'pass' if should_pass else 'fail'} | Got: {'pass' if passed else 'fail'} → {expectation}\n")

        results.append({
            "scenario": scenario,
            "args": args,
            "should_pass": should_pass,
            "passed": passed,
            "violation": violation,
            "correctly_classified": correct,
        })

    return results


def run_llm_experiment(client: OpenAI) -> list[dict]:
    """
    Ask the LLM to generate tool calls and test them against the contract.
    This simulates what a real function-calling pipeline would look like.
    """
    PROMPTS = [
        "Book a return flight from London to New York, 2 adults, economy, departing June 1st returning June 15th 2025.",
        "Book a flight returning before it departs — from Paris to LA, depart July 10th, return July 5th 2025, 1 passenger, business.",
        "I need 3 infant seats on a flight from Dubai to Mumbai, no adult seats, economy, September 1st 2025.",
    ]

    results = []
    for prompt in PROMPTS:
        console.print(f"\n[bold]LLM prompt:[/] {prompt}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a flight booking assistant. "
                        "Respond ONLY with a JSON object representing the tool call arguments for book_flight. "
                        "Fields: origin (IATA), destination (IATA), depart_date (YYYY-MM-DD), "
                        "return_date (YYYY-MM-DD, optional), passengers (int), "
                        "infant_passengers (int, optional), cabin_class (economy/business/first). "
                        "Respond with raw JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        import re
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            args = json.loads(clean)
        except json.JSONDecodeError:
            console.print(f"[red]Could not parse LLM output as JSON: {raw}[/]")
            continue

        console.print(f"[dim]LLM produced:[/] {json.dumps(args)}")
        try:
            FLIGHT_CONTRACT.validate(args)
            console.print("[green]Contract: PASS — would execute[/]")
            results.append({"prompt": prompt, "args": args, "passed": True, "violation": None})
        except ToolContractViolation as e:
            console.print(f"[red]Contract: FAIL — blocked before execution[/]")
            console.print(f"[red]  Rule: {e.rule}[/]")
            console.print(f"[red]  Detail: {e.detail}[/]")
            results.append({"prompt": prompt, "args": args, "passed": False, "violation": str(e)})

    return results


def print_summary(results: list[dict]) -> None:
    table = Table(title="\nTool Contract Test Summary", show_lines=True)
    table.add_column("Scenario", min_width=40)
    table.add_column("Expected", justify="center")
    table.add_column("Got", justify="center")
    table.add_column("Correct?", justify="center")

    for r in results:
        table.add_row(
            r["scenario"],
            "pass" if r["should_pass"] else "fail",
            "pass" if r["passed"] else "fail",
            "[green]✓[/]" if r["correctly_classified"] else "[red]✗[/]",
        )
    console.print(table)

    correct = sum(1 for r in results if r["correctly_classified"])
    console.print(
        f"\n[bold]{correct}/{len(results)} test cases correctly classified.[/]\n"
        "Semantic rules caught violations that JSON schema alone would have missed.\n"
    )


if __name__ == "__main__":
    results = run_experiment()
    print_summary(results)

    # Optionally run LLM experiment
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        console.print("\n[bold cyan]Running LLM-generated tool call experiment...[/]")
        client = OpenAI(api_key=api_key)
        llm_results = run_llm_experiment(client)
    else:
        console.print("\n[dim]Skipping LLM experiment (OPENAI_API_KEY not set)[/]")
        llm_results = []

    out_path = Path(__file__).parent.parent / "results" / "tool_violations.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"static_tests": results, "llm_tests": llm_results}, f, indent=2)
    console.print(f"\nResults saved to [cyan]{out_path}[/]")
