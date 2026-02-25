"""
experiments/retrieval_instability.py

Experiment 2: Retrieval Instability After Index Changes
=======================================================

Hypothesis: Retrieval systems silently change *which* sources they surface when
chunking strategy or corpus is updated, even when standard metrics (recall, latency)
appear stable. Without explicit drift contracts, this is invisible.

Method:
  We simulate two retrieval snapshots for the same query:
  - Baseline corpus: a set of "approved" documents (known grounding sources)
  - Updated corpus: a re-chunked or refreshed version that drifts to new sources

  The RetrievalContract detects the drift by computing Jaccard overlap between
  source sets. It also enforces source allowlists.

  We use OpenAI embeddings to build a mini in-memory vector store
  and run nearest-neighbour retrieval against both corpora.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from contracts.retrieval import RetrievalContract, RetrievalContractViolation, RetrievedChunk

load_dotenv()
console = Console()

# ---------------------------------------------------------------------------
# Simulated corpus — baseline
# ---------------------------------------------------------------------------
BASELINE_CORPUS = [
    {"id": "policy_v1_sec1", "text": "The company's data retention policy requires all user data to be deleted after 90 days of inactivity."},
    {"id": "policy_v1_sec2", "text": "Access logs must be kept for a minimum of 12 months for compliance purposes."},
    {"id": "policy_v1_sec3", "text": "Personal data may only be processed with explicit user consent under GDPR Article 6."},
    {"id": "policy_v1_sec4", "text": "Data breaches must be reported to the supervisory authority within 72 hours."},
    {"id": "policy_v1_sec5", "text": "Encryption at rest is mandatory for all databases containing personally identifiable information."},
]

# Updated corpus — different chunking strategy introduces new source IDs
# and drops some original ones. Standard recall metrics might look fine,
# but the grounding sources have shifted.
UPDATED_CORPUS = [
    {"id": "policy_v2_chunk_a", "text": "User data deletion: accounts inactive for 90+ days trigger automatic data removal per retention schedule."},
    {"id": "policy_v2_chunk_b", "text": "Audit logs including access logs are retained for 12 months minimum to satisfy regulatory requirements."},
    {"id": "policy_v2_chunk_c", "text": "GDPR consent requirements: Article 6 mandates lawful basis for processing. Consent is one valid basis."},
    {"id": "external_blog_post", "text": "Data privacy best practices suggest quarterly reviews of retention schedules and consent flows."},  # new external source!
    {"id": "draft_policy_v3",   "text": "DRAFT: Proposed new policy would extend retention to 180 days pending legal review. NOT YET APPROVED."},  # dangerous drift!
]

QUERY = "What are our data retention and deletion obligations?"

CONTRACT = RetrievalContract(
    allowed_sources={
        "policy_v1_sec1", "policy_v1_sec2", "policy_v1_sec3",
        "policy_v1_sec4", "policy_v1_sec5",
        "policy_v2_chunk_a", "policy_v2_chunk_b", "policy_v2_chunk_c",
    },
    blocked_sources={"draft_policy_v3"},  # drafts must never be surfaced
    min_results=2,
    max_results=5,
    min_jaccard_overlap=0.5,  # require 50% source overlap between baseline and updated
)


def embed(client: OpenAI, texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    response = client.embeddings.create(input=texts, model=model)
    return np.array([e.embedding for e in response.data])


def retrieve_top_k(
    query_vec: np.ndarray,
    corpus_vecs: np.ndarray,
    corpus: list[dict],
    k: int = 3,
) -> list[RetrievedChunk]:
    sims = corpus_vecs @ query_vec / (
        np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    top_k_idx = np.argsort(sims)[::-1][:k]
    return [
        RetrievedChunk(
            source_id=corpus[i]["id"],
            content=corpus[i]["text"],
            score=float(sims[i]),
            retrieved_at=datetime.utcnow(),
        )
        for i in top_k_idx
    ]


def run_experiment(client: OpenAI) -> dict:
    console.print("\n[bold cyan]Embedding corpora...[/]")
    query_vec = embed(client, [QUERY])[0]

    baseline_vecs = embed(client, [d["text"] for d in BASELINE_CORPUS])
    updated_vecs = embed(client, [d["text"] for d in UPDATED_CORPUS])

    baseline_chunks = retrieve_top_k(query_vec, baseline_vecs, BASELINE_CORPUS, k=3)
    updated_chunks = retrieve_top_k(query_vec, updated_vecs, UPDATED_CORPUS, k=3)

    console.print("\n[bold]Baseline retrieval results:[/]")
    for c in baseline_chunks:
        console.print(f"  [{c.score:.3f}] [cyan]{c.source_id}[/] — {c.content[:80]}...")

    console.print("\n[bold]Updated retrieval results:[/]")
    for c in updated_chunks:
        console.print(f"  [{c.score:.3f}] [cyan]{c.source_id}[/] — {c.content[:80]}...")

    results = {"baseline_passed": None, "updated_passed": None, "drift_passed": None,
                "baseline_violation": None, "updated_violation": None, "drift_violation": None,
                "jaccard_score": None}

    # Check 1: Baseline passes the contract
    console.print("\n[bold cyan]Running contract checks...[/]")
    try:
        CONTRACT.validate(baseline_chunks)
        console.print("[green]✓ Baseline: PASS[/]")
        results["baseline_passed"] = True
    except RetrievalContractViolation as e:
        console.print(f"[red]✗ Baseline: FAIL — {e}[/]")
        results["baseline_passed"] = False
        results["baseline_violation"] = str(e)

    # Check 2: Updated corpus — contains blocked source
    try:
        CONTRACT.validate(updated_chunks)
        console.print("[green]✓ Updated: PASS[/]")
        results["updated_passed"] = True
    except RetrievalContractViolation as e:
        console.print(f"[red]✗ Updated: FAIL — {e}[/]")
        results["updated_passed"] = False
        results["updated_violation"] = str(e)

    # Check 3: Drift detection
    try:
        score = CONTRACT.check_drift(baseline_chunks, updated_chunks)
        console.print(f"[green]✓ Drift check: PASS (Jaccard={score:.2f})[/]")
        results["drift_passed"] = True
        results["jaccard_score"] = score
    except RetrievalContractViolation as e:
        console.print(f"[red]✗ Drift check: FAIL — {e}[/]")
        results["drift_passed"] = False
        results["drift_violation"] = str(e)

    return results


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not set.[/]")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    results = run_experiment(client)

    console.print("\n[bold]Summary:[/]")
    console.print(f"  Baseline contract: {'✓ PASS' if results['baseline_passed'] else '✗ FAIL'}")
    console.print(f"  Updated contract:  {'✓ PASS' if results['updated_passed'] else '✗ FAIL'} "
                  f"{'(caught blocked source!)' if not results['updated_passed'] else ''}")
    console.print(f"  Drift check:       {'✓ PASS' if results['drift_passed'] else '✗ FAIL'} "
                  f"(Jaccard={results['jaccard_score']:.2f})" if results['jaccard_score'] else "")

    out_path = Path(__file__).parent.parent / "results" / "retrieval_instability.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nResults saved to [cyan]{out_path}[/]")
