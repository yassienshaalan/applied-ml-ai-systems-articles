"""
experiments/embedding_upgrade.py

Experiment 3: Embedding Upgrades as Architectural Changes
==========================================================

Hypothesis: Swapping embedding models is an architectural change, not an
implementation detail. It silently shifts the geometry of the meaning space,
altering similarity rankings and neighbourhood structure in ways that break
downstream retrieval — even if code is unchanged.

Method:
  1. Embed a shared corpus with model v1 (text-embedding-3-small).
  2. Embed the same corpus with model v2 (text-embedding-ada-002).
  3. Apply EmbeddingContract checks:
     - Dimension check: dimensions may differ across models.
     - Cross-similarity: mean cosine similarity between corresponding vectors.
     - Neighbourhood stability: do k-NN rankings change significantly?
  4. Show that the contract catches the upgrade as a breaking change.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from contracts.embedding import EmbeddingContract, EmbeddingContractViolation

load_dotenv()
console = Console()

CORPUS = [
    "The company's data retention policy requires deletion after 90 days of inactivity.",
    "Access logs must be kept for a minimum of 12 months for compliance.",
    "Personal data may only be processed with explicit user consent.",
    "Data breaches must be reported within 72 hours.",
    "Encryption at rest is mandatory for all PII databases.",
    "Machine learning models trained on customer data require a privacy impact assessment.",
    "Third-party data processors must sign a data processing agreement.",
    "Users have the right to request deletion of their personal data.",
    "The legal basis for marketing emails is opt-in consent.",
    "Anonymised data is exempt from GDPR data subject rights.",
]

MODEL_V1 = "text-embedding-3-small"   # 1536 dimensions
MODEL_V2 = "text-embedding-ada-002"   # 1536 dimensions — same dims, different geometry!

# Contract for v1 model
CONTRACT_V1 = EmbeddingContract(
    expected_dimensions=1536,
    min_mean_cross_similarity=0.90,     # strict: same text should map similarly
    min_neighbourhood_overlap=0.70,     # 70% of neighbours should be stable
)


def embed(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    console.print(f"  Embedding {len(texts)} texts with [cyan]{model}[/]...")
    response = client.embeddings.create(input=texts, model=model)
    return [e.embedding for e in response.data]


def run_experiment(client: OpenAI) -> dict:
    console.print("\n[bold cyan]Step 1: Embed corpus with v1 model[/]")
    v1_embeddings = embed(client, CORPUS, MODEL_V1)

    console.print("[bold cyan]Step 2: Embed same corpus with v2 model (the 'upgrade')[/]")
    v2_embeddings = embed(client, CORPUS, MODEL_V2)

    results = {
        "model_v1": MODEL_V1,
        "model_v2": MODEL_V2,
        "corpus_size": len(CORPUS),
        "dim_v1": len(v1_embeddings[0]),
        "dim_v2": len(v2_embeddings[0]),
        "dimension_check_passed": None,
        "distributional_check_passed": None,
        "neighbourhood_check_passed": None,
        "cross_similarity_score": None,
        "neighbourhood_overlap_score": None,
        "violations": [],
    }

    console.print("\n[bold cyan]Step 3: Apply contract checks[/]")

    # Check dimensions of v2 against v1 contract
    try:
        CONTRACT_V1.check_dimensions(v2_embeddings)
        console.print(f"[green]✓ Dimension check: PASS (v2 dims = {len(v2_embeddings[0])})[/]")
        results["dimension_check_passed"] = True
    except EmbeddingContractViolation as e:
        console.print(f"[red]✗ Dimension check: FAIL — {e}[/]")
        results["dimension_check_passed"] = False
        results["violations"].append(str(e))

    # Distributional stability
    try:
        score = CONTRACT_V1.check_distributional_stability(v1_embeddings, v2_embeddings)
        console.print(f"[green]✓ Distributional stability: PASS (mean cosine = {score:.4f})[/]")
        results["distributional_check_passed"] = True
        results["cross_similarity_score"] = score
    except EmbeddingContractViolation as e:
        # Compute score even on failure for reporting
        v1_arr = np.array(v1_embeddings)
        v2_arr = np.array(v2_embeddings)
        sims = [
            float(np.dot(v1_arr[i], v2_arr[i]) / (np.linalg.norm(v1_arr[i]) * np.linalg.norm(v2_arr[i]) + 1e-10))
            for i in range(len(v1_embeddings))
        ]
        score = float(np.mean(sims))
        console.print(f"[red]✗ Distributional stability: FAIL (mean cosine = {score:.4f}, min={CONTRACT_V1.min_mean_cross_similarity})[/]")
        console.print(f"[red]  → {e}[/]")
        results["distributional_check_passed"] = False
        results["cross_similarity_score"] = score
        results["violations"].append(str(e))

    # Neighbourhood stability
    try:
        overlap = CONTRACT_V1.check_neighbourhood_stability(v1_embeddings, v2_embeddings, k=3)
        console.print(f"[green]✓ Neighbourhood stability: PASS (mean overlap = {overlap:.4f})[/]")
        results["neighbourhood_check_passed"] = True
        results["neighbourhood_overlap_score"] = overlap
    except EmbeddingContractViolation as e:
        # Compute overlap even on failure
        v1_arr = np.array(v1_embeddings)
        v2_arr = np.array(v2_embeddings)

        def knn(matrix, i, k):
            q = matrix[i]
            sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-10)
            sims[i] = -1
            return set(np.argsort(sims)[-k:].tolist())

        overlaps = []
        for i in range(len(v1_embeddings)):
            b = knn(v1_arr, i, 3)
            u = knn(v2_arr, i, 3)
            inter = b & u
            uni = b | u
            overlaps.append(len(inter) / len(uni) if uni else 1.0)
        overlap = float(np.mean(overlaps))

        console.print(f"[red]✗ Neighbourhood stability: FAIL (mean overlap = {overlap:.4f}, min={CONTRACT_V1.min_neighbourhood_overlap})[/]")
        console.print(f"[red]  → This means retrieval rankings change significantly after the upgrade.[/]")
        results["neighbourhood_check_passed"] = False
        results["neighbourhood_overlap_score"] = overlap
        results["violations"].append(str(e))

    return results


def print_summary(results: dict) -> None:
    table = Table(title="\nEmbedding Upgrade Experiment — Contract Results", show_lines=True)
    table.add_column("Check")
    table.add_column("Result", justify="center")
    table.add_column("Score")
    table.add_column("Threshold")

    table.add_row(
        "Dimension compatibility",
        "[green]✓[/]" if results["dimension_check_passed"] else "[red]✗[/]",
        f"{results['dim_v1']} → {results['dim_v2']}",
        f"{CONTRACT_V1.expected_dimensions}",
    )
    table.add_row(
        "Distributional stability",
        "[green]✓[/]" if results["distributional_check_passed"] else "[red]✗[/]",
        f"{results['cross_similarity_score']:.4f}" if results["cross_similarity_score"] else "N/A",
        f"≥ {CONTRACT_V1.min_mean_cross_similarity}",
    )
    table.add_row(
        "Neighbourhood stability",
        "[green]✓[/]" if results["neighbourhood_check_passed"] else "[red]✗[/]",
        f"{results['neighbourhood_overlap_score']:.4f}" if results["neighbourhood_overlap_score"] else "N/A",
        f"≥ {CONTRACT_V1.min_neighbourhood_overlap}",
    )
    console.print(table)

    violations = results["violations"]
    if violations:
        console.print(
            f"\n[bold red]Embedding upgrade detected as architectural change ({len(violations)} violation(s)).[/]\n"
            "[bold]Required actions:[/] re-baseline retrieval indices, validate downstream queries, versioned migration.\n"
        )
    else:
        console.print("\n[bold green]Embedding upgrade is contract-compatible. Safe to proceed.[/]\n")


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not set.[/]")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    results = run_experiment(client)
    print_summary(results)

    out_path = Path(__file__).parent.parent / "results" / "embedding_upgrade.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"Results saved to [cyan]{out_path}[/]")
