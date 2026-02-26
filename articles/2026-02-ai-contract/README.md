# AI Contracts: Engineering Boundaries in Systems That Generate

This folder contains the code, experiments, and supporting material for the article:

**"AI Contracts: Engineering Boundaries in Systems That Generate"**

The article argues that generative AI systems fail not dramatically but silently like prompts drift, retrieval corpora shift, embedding models are swapped, and tool calls encode impossible operations, all without triggering alerts. It proposes a contract-first architecture where explicit, testable boundaries are enforced at each generative interface, in the same way mature software systems enforce contracts at APIs and services.

---

## Article Link

Medium article:
[AI Contracts: Engineering Boundaries in Systems That Generate](https://medium.com/p/21a8e294c813)

---

## What This Article Is About

As generative AI systems have become more capable, their failure modes have changed:

- Prompts behave like APIs but have no schemas, so minor refactors silently break downstream behaviour.
- Retrieval systems are monitored for latency and recall, but not for *which sources* they rely on, allowing grounding to drift invisibly.
- Embedding model upgrades are treated as implementation details, when they are in fact architectural changes that shift the geometry of the entire meaning space.
- Tool calls are validated structurally (valid JSON) but not semantically, allowing logically impossible operations to pass silently into execution.

The question is not whether these systems fail, but whether failures are made visible at the boundary where they occur, or allowed to accumulate silently until they surface as degraded behaviour in production.

## Why This Matters

The article's insights apply to:

- **AI system reliability:** where silent drift in prompts, retrieval, or embeddings erodes trustworthiness without triggering any observable signal.
- **Engineering governance:** because contracts make generative system behaviour inspectable, testable, and auditable across the delivery pipeline.
- **CI/CD for AI:** treating prompt changes, corpus refreshes, and model upgrades as first-class architectural events that must pass explicit contract checks before deployment.

By shifting focus from model capability to system boundaries, teams can build generative systems that are not only powerful but reliably governable under change.

---

## Empirical Validation Suite

This folder contains an experimental test suite for the hypothesis that **explicit contracts at generative AI system boundaries catch silent regressions** that traditional monitoring misses.

It mirrors the four failure modes described in the article:

1. Prompt drift under minor refactoring
2. Retrieval instability after index changes
3. Embedding upgrades as architectural changes
4. Semantic violations in tool interactions

### Architecture

```
2026-02-ai-contract/
├── contracts/               # Contract definitions (the control plane)
│   ├── __init__.py
│   ├── prompt.py            # Output schema + behavioural invariants
│   ├── retrieval.py         # Source stability + drift thresholds
│   ├── embedding.py         # Representation compatibility checks
│   └── tool.py              # Schema + semantic validation
├── experiments/             # LLM interaction harnesses
│   ├── prompt_drift.py
│   ├── retrieval_instability.py
│   ├── embedding_upgrade.py
│   └── tool_violations.py
├── tests/                   # Pytest suites — the CI/CD contract pack
│   ├── test_prompt_contracts.py
│   ├── test_retrieval_contracts.py
│   ├── test_embedding_contracts.py
│   └── test_tool_contracts.py
├── results/                 # Saved experiment outputs (JSON)
├── run_tests.py             # Lightweight runner — no pytest required
├── runner.py                # Run all experiments and print a summary report
└── requirements.txt
```

### Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

### Running

```bash
# Offline tests — no API key needed, runs in ~1 second
python run_tests.py

# Run a single live experiment
python experiments/prompt_drift.py
python experiments/retrieval_instability.py
python experiments/embedding_upgrade.py
python experiments/tool_violations.py

# Run all experiments and print a consolidated report
python runner.py
```

### How contracts work

Each contract is a plain Python dataclass with a `.validate()` or `.check()` method that raises a typed violation exception on failure. Tests import contracts and assert violations are raised (or not raised) under controlled conditions.

The key insight: contracts run *outside* the model, at the system boundary. The LLM stays flexible; the surrounding architecture enforces structure.

## Experiment Results

The `results/` folder contains the raw JSON output from each experiment run.

- **`prompt_drift.json`** : responses from all three prompt variants (production, tone refactor, brevity refactor) with contract pass/fail status and violation details. 1/3 passed; both refactors failed on missing required fields.
- **`retrieval_instability.json`** : baseline vs updated retrieval sources, Jaccard overlap score, and which contract checks fired. Jaccard dropped to 0.00 — zero source overlap between the two corpus snapshots — with an external blog post surfacing in top results.
- **`tool_violations.json`** : static test classifications plus live LLM-generated tool calls showing what the contract blocked before execution. 7/7 correctly classified; the LLM faithfully generated an impossible return-before-departure booking which was caught at the semantic rule boundary.
- **`embedding_upgrade.json`** : dimension check, cross-similarity (-0.0062), and neighbourhood overlap (0.85) from comparing `text-embedding-3-small` vs `text-embedding-ada-002` on the same corpus. Both models output 1536 dimensions  structurally identical but near-zero cross-similarity flags the upgrade as an architectural change. Neighbourhood stability partially holds, suggesting relative document ordering is more robust to model swaps than absolute vector representations.

These are the empirical outputs the article's conclusions are drawn from.
