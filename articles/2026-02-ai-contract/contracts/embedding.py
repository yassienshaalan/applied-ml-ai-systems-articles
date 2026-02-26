"""
contracts/embedding.py — Embedding Contract (pure Python + numpy)

Treats embedding upgrades as architectural events, not implementation details.
Enforces dimension compatibility, distributional stability, and neighbourhood stability.
"""

ffrom __future__ import annotations
from dataclasses import dataclass
import numpy as np


class EmbeddingContractViolation(Exception):
    def __init__(self, rule: str, detail: str):
        self.rule = rule
        self.detail = detail
        super().__init__(f"EmbeddingContract violation [{rule}]: {detail}")


def _cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


@dataclass
class EmbeddingContract:
    expected_dimensions: int
    min_mean_cross_similarity: float = 0.85
    min_neighbourhood_overlap: float = 0.6
    min_self_similarity: float = 0.99

    def check_dimensions(self, embeddings: list[list[float]]) -> None:
        for i, emb in enumerate(embeddings):
            if len(emb) != self.expected_dimensions:
                raise EmbeddingContractViolation("expected_dimensions",
                    f"Embedding[{i}] has {len(emb)} dims, expected {self.expected_dimensions}.")

    def check_distributional_stability(
        self, baseline: list[list[float]], updated: list[list[float]]
    ) -> float:
        if len(baseline) != len(updated):
            raise EmbeddingContractViolation("distributional_stability",
                f"Baseline has {len(baseline)} embeddings but updated has {len(updated)}.")
        sims = [_cosine_sim(np.array(a), np.array(b)) for a, b in zip(baseline, updated)]
        score = float(np.mean(sims))
        if score < self.min_mean_cross_similarity:
            raise EmbeddingContractViolation("min_mean_cross_similarity",
                f"Mean cross-similarity {score:.4f} < minimum {self.min_mean_cross_similarity:.4f}. "
                "Treat embedding upgrade as an architectural change requiring re-baselining.")
        return score

    def check_neighbourhood_stability(
        self, baseline: list[list[float]], updated: list[list[float]], k: int = 5
    ) -> float:
        n = len(baseline)
        if n < k + 1:
            return 1.0
        ba = np.array(baseline, dtype=np.float64)
        ua = np.array(updated, dtype=np.float64)

        def knn(matrix, i, k):
            q = matrix[i]
            sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-10)
            sims[i] = -1
            return set(np.argsort(sims)[-k:].tolist())

        overlaps = []
        for i in range(n):
            b_nb = knn(ba, i, k)
            u_nb = knn(ua, i, k)
            inter = b_nb & u_nb
            union = b_nb | u_nb
            overlaps.append(len(inter) / len(union) if union else 1.0)
        mean_overlap = float(np.mean(overlaps))
        if mean_overlap < self.min_neighbourhood_overlap:
            raise EmbeddingContractViolation("min_neighbourhood_overlap",
                f"Mean neighbourhood overlap {mean_overlap:.4f} < minimum {self.min_neighbourhood_overlap:.4f}. "
                f"k={k}. Retrieval rankings will change significantly — re-baseline required.")
        return mean_overlap