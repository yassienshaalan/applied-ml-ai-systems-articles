# AI Contracts: Empirical Validation Suite

Experimental test suite for the hypothesis that **explicit contracts at generative AI system boundaries catch silent regressions** that traditional monitoring misses.

Mirrors the four failure modes described in the article:
1. Prompt drift under minor refactoring
2. Retrieval instability after index changes
3. Embedding upgrades as architectural changes
4. Semantic violations in tool interactions

## Architecture

```
ai-contracts/
├── contracts/           # Pydantic contract definitions (the control plane)
│   ├── prompt.py        # Output schema + behavioural invariants
│   ├── retrieval.py     # Source stability + drift thresholds
│   ├── embedding.py     # Representation compatibility checks
│   └── tool.py          # Schema + semantic validation
├── experiments/         # LLM interaction harnesses
│   ├── prompt_drift.py
│   ├── retrieval_instability.py
│   ├── embedding_upgrade.py
│   └── tool_violations.py
├── tests/               # Pytest suites — the CI/CD contract pack
│   ├── test_prompt_contracts.py
│   ├── test_retrieval_contracts.py
│   ├── test_embedding_contracts.py
│   └── test_tool_contracts.py
├── results/             # Saved experiment outputs (JSON)
├── conftest.py          # Shared fixtures (OpenAI client, models)
├── runner.py            # Run all experiments and print a summary report
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Running experiments

```bash
# Run all contract tests (CI/CD contract pack)
pytest tests/ -v

# Run a single experiment interactively and see detailed output
python experiments/prompt_drift.py
python experiments/retrieval_instability.py
python experiments/embedding_upgrade.py
python experiments/tool_violations.py

# Run all experiments and generate a summary report
python runner.py
```

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable baseline |
| `experiments/ai-contracts-validation` | Active experiment work |

## How contracts work

Each contract is a Pydantic model with a `.validate_response()` or `.check()` method that raises `ContractViolationError` on failure. Tests import contracts and assert violations are raised (or not raised) under controlled conditions.

The key insight: contracts run *outside* the model, at the system boundary. The LLM stays flexible; the surrounding architecture enforces structure.
