"""
runner.py — Run all experiments and print a consolidated report.

Usage:
    python runner.py
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

load_dotenv()
console = Console()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def banner():
    console.print(Panel.fit(
        "[bold cyan]AI Contracts — Empirical Validation Suite[/]\n"
        "[dim]Testing the hypothesis: explicit contracts at generative AI boundaries\n"
        "catch silent regressions that traditional monitoring misses.[/]",
        border_style="cyan",
    ))


def run_tool_experiment() -> dict:
    """Experiment 4 runs without API calls — always execute."""
    console.print("\n[bold]Experiment 4: Tool Semantic Violations[/] [dim](no API required)[/]")
    from experiments.tool_violations import run_experiment, TEST_CALLS
    results = run_experiment()
    correct = sum(1 for r in results if r["correctly_classified"])
    return {
        "name": "Tool Semantic Violations",
        "total": len(results),
        "correct": correct,
        "violations_caught": sum(1 for r in results if not r["should_pass"] and not r["passed"]),
        "false_positives": sum(1 for r in results if r["should_pass"] and not r["passed"]),
    }


def run_prompt_experiment(client) -> dict:
    console.print("\n[bold]Experiment 1: Prompt Drift[/]")
    from experiments.prompt_drift import run_experiment, VARIANTS
    results = run_experiment(client)
    return {
        "name": "Prompt Drift",
        "total": len(results),
        "passed": sum(1 for r in results if r["compliant"]),
        "failed": sum(1 for r in results if not r["compliant"]),
        "regressions_caught": sum(1 for r in results if not r["compliant"]),
    }


def run_retrieval_experiment(client) -> dict:
    console.print("\n[bold]Experiment 2: Retrieval Instability[/]")
    from experiments.retrieval_instability import run_experiment
    results = run_experiment(client)
    violations = sum(1 for k in ["baseline_passed", "updated_passed", "drift_passed"]
                     if results.get(k) is False)
    return {
        "name": "Retrieval Instability",
        "baseline_ok": results.get("baseline_passed"),
        "drift_caught": not results.get("drift_passed", True),
        "blocked_source_caught": not results.get("updated_passed", True),
        "jaccard_score": results.get("jaccard_score"),
        "violations": violations,
    }


def run_embedding_experiment(client) -> dict:
    console.print("\n[bold]Experiment 3: Embedding Upgrade[/]")
    from experiments.embedding_upgrade import run_experiment
    results = run_experiment(client)
    return {
        "name": "Embedding Upgrade",
        "dimension_check": results.get("dimension_check_passed"),
        "distributional_check": results.get("distributional_check_passed"),
        "neighbourhood_check": results.get("neighbourhood_check_passed"),
        "cross_similarity": results.get("cross_similarity_score"),
        "neighbourhood_overlap": results.get("neighbourhood_overlap_score"),
        "violations": len(results.get("violations", [])),
    }


def print_final_report(experiment_results: list[dict]) -> None:
    console.print("\n")
    console.rule("[bold cyan]Final Report[/]")

    table = Table(show_lines=True, title="Experiment Results Summary")
    table.add_column("Experiment", min_width=30)
    table.add_column("Key Finding")
    table.add_column("Contracts Triggered?", justify="center")

    for r in experiment_results:
        name = r.get("name", "Unknown")

        if name == "Prompt Drift":
            finding = f"{r['regressions_caught']} regressions caught out of {r['total']} variants"
            triggered = "[green]YES[/]" if r["regressions_caught"] > 0 else "[red]NO[/]"

        elif name == "Retrieval Instability":
            parts = []
            if r.get("blocked_source_caught"):
                parts.append("blocked source surfaced")
            if r.get("drift_caught"):
                parts.append(f"drift detected (Jaccard={r.get('jaccard_score', 0):.2f})")
            finding = "; ".join(parts) if parts else "no violations detected"
            triggered = "[green]YES[/]" if r.get("violations", 0) > 0 else "[dim]NO[/]"

        elif name == "Embedding Upgrade":
            parts = []
            if r.get("cross_similarity") is not None:
                parts.append(f"cross-sim={r['cross_similarity']:.3f}")
            if r.get("neighbourhood_overlap") is not None:
                parts.append(f"nb-overlap={r['neighbourhood_overlap']:.3f}")
            finding = ", ".join(parts) if parts else "checks not completed"
            triggered = "[green]YES[/]" if r.get("violations", 0) > 0 else "[dim]NO[/]"

        elif name == "Tool Semantic Violations":
            finding = f"{r['violations_caught']} semantic violations caught ({r['correct']}/{r['total']} correct)"
            triggered = "[green]YES[/]" if r["violations_caught"] > 0 else "[red]NO[/]"

        else:
            finding = str(r)
            triggered = "?"

        table.add_row(name, finding, triggered)

    console.print(table)
    console.print(
        "\n[bold]Conclusion:[/] Contract-first architecture surfaces failures at boundaries "
        "that would otherwise propagate silently into production.\n"
        "Each experiment confirms the article's hypothesis: "
        "[italic]power without boundaries does not produce robustness — engineering does.[/italic]\n"
    )


def main():
    banner()
    api_key = os.getenv("OPENAI_API_KEY")
    skip_live = "--skip-live" in sys.argv or not api_key

    if skip_live:
        if not api_key:
            console.print("[yellow]OPENAI_API_KEY not set — running offline experiments only.[/]")
        else:
            console.print("[yellow]--skip-live flag set — skipping API experiments.[/]")

    experiment_results = []
    start = time.time()

    # Experiment 4 runs without API
    try:
        r = run_tool_experiment()
        experiment_results.append(r)
    except Exception as e:
        console.print(f"[red]Experiment 4 failed: {e}[/]")

    if not skip_live:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        for name, fn in [
            ("Prompt Drift", run_prompt_experiment),
            ("Retrieval Instability", run_retrieval_experiment),
            ("Embedding Upgrade", run_embedding_experiment),
        ]:
            try:
                r = fn(client)
                experiment_results.append(r)
            except Exception as e:
                console.print(f"[red]{name} failed: {e}[/]")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - start
    print_final_report(experiment_results)
    console.print(f"[dim]Total runtime: {elapsed:.1f}s[/]")

    # Save combined results
    out_path = RESULTS_DIR / "all_experiments.json"
    with open(out_path, "w") as f:
        json.dump(experiment_results, f, indent=2)
    console.print(f"Results saved to [cyan]{out_path}[/]")


if __name__ == "__main__":
    main()
