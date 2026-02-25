"""
run_tests.py — Lightweight test runner (no pytest required).

Run with: python run_tests.py
For live API tests: python run_tests.py --live

If you have pytest installed, you can also run: pytest tests/ -v [--live]
"""

import sys, os, traceback, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m○\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = {"passed": 0, "failed": 0, "skipped": 0}
failures = []

def expect_raises(exc_class, fn, *args, rule=None, **kwargs):
    try:
        fn(*args, **kwargs)
        return False, f"Expected {exc_class.__name__} but nothing was raised"
    except exc_class as e:
        if rule and hasattr(e, 'rule') and e.rule != rule:
            return False, f"Expected rule='{rule}' but got rule='{e.rule}'"
        return True, ""
    except Exception as e:
        return False, f"Wrong exception: {type(e).__name__}: {e}"

def expect_no_raise(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return True, ""
    except Exception as e:
        return False, f"Unexpected exception: {type(e).__name__}: {e}"

def test(name, ok, reason=""):
    global results
    if ok:
        results["passed"] += 1
        print(f"  {PASS} {name}")
    else:
        results["failed"] += 1
        failures.append((name, reason))
        print(f"  {FAIL} {name}")
        if reason:
            print(f"       {reason}")

def section(title):
    print(f"\n{BOLD}{title}{RESET}")

# ===========================================================================
# PROMPT CONTRACT TESTS
# ===========================================================================
section("Prompt Contract")

from contracts.prompt import PromptContract, PromptContractViolation
import json as _json

contract = PromptContract(
    required_fields=["answer", "confidence", "sources", "reasoning"],
    forbidden_strings=["I don't know", "As an AI"],
    must_cite_sources=True,
    allowed_confidence_values=["high", "medium", "low"],
    output_format="json",
    max_tokens=400,
)

valid = _json.dumps({"answer": "Paris", "confidence": "high",
                      "sources": ["Britannica"], "reasoning": "Historical."})

ok, reason = expect_no_raise(contract.validate_response, valid)
test("Valid response passes", ok, reason)

fenced = '```json\n{"answer":"x","confidence":"low","sources":["s"],"reasoning":"r"}\n```'
ok, reason = expect_no_raise(contract.validate_response, fenced)
test("Markdown-fenced JSON parsed correctly", ok, reason)

missing_sources = _json.dumps({"answer": "Paris", "confidence": "high"})
ok, reason = expect_raises(PromptContractViolation, contract.validate_response, missing_sources)
test("Missing required field raises violation", ok, reason)

forbidden = _json.dumps({"answer": "As an AI I'm not sure", "confidence": "low",
                          "sources": ["x"], "reasoning": "y"})
ok, reason = expect_raises(PromptContractViolation, contract.validate_response, forbidden,
                            rule="forbidden_string")
test("Forbidden string raises violation", ok, reason)

bad_conf = _json.dumps({"answer": "Paris", "confidence": "95%",
                         "sources": ["x"], "reasoning": "y"})
ok, reason = expect_raises(PromptContractViolation, contract.validate_response, bad_conf,
                            rule="allowed_confidence_values")
test("Invalid confidence value raises violation", ok, reason)

empty_sources = _json.dumps({"answer": "Paris", "confidence": "high",
                               "sources": [], "reasoning": "y"})
ok, reason = expect_raises(PromptContractViolation, contract.validate_response, empty_sources,
                            rule="must_cite_sources")
test("Empty sources raises violation", ok, reason)

ok, reason = expect_raises(PromptContractViolation, contract.validate_response,
                            "not json at all", rule="json_parse")
test("Non-JSON raises violation", ok, reason)

# Drift simulation
v2 = _json.dumps({"response": "Paris!", "confidence_level": 95,
                   "sources": ["Wikipedia"], "reasoning": "Known fact."})
compliant, _ = contract.is_compliant(v2)
test("Tone-refactor (renamed fields) caught as violation", not compliant)

v3 = _json.dumps({"answer": "Paris", "confidence": "high"})
compliant, _ = contract.is_compliant(v3)
test("Brevity-refactor (dropped fields) caught as violation", not compliant)

# ===========================================================================
# RETRIEVAL CONTRACT TESTS
# ===========================================================================
section("Retrieval Contract")

from contracts.retrieval import RetrievalContract, RetrievalContractViolation, RetrievedChunk
from datetime import datetime, timedelta

def chunk(source_id, score=0.9, age_days=0):
    return RetrievedChunk(source_id=source_id, content=f"text from {source_id}",
                          score=score, retrieved_at=datetime.utcnow() - timedelta(days=age_days))

rc = RetrievalContract(
    allowed_sources={"doc_a", "doc_b", "doc_c", "doc_d"},
    blocked_sources={"draft_v3"},
    min_results=2, max_results=5,
    min_jaccard_overlap=0.5, max_age_days=30,
)

baseline = [chunk("doc_a"), chunk("doc_b"), chunk("doc_c")]
ok, reason = expect_no_raise(rc.validate, baseline)
test("Allowed sources pass validation", ok, reason)

ok, reason = expect_raises(RetrievalContractViolation, rc.validate,
                            [chunk("doc_a"), chunk("unknown_wiki")], rule="allowed_sources")
test("Disallowed source raises violation", ok, reason)

ok, reason = expect_raises(RetrievalContractViolation, rc.validate,
                            [chunk("doc_a"), chunk("draft_v3")])
test("Blocked source raises violation", ok, reason)

ok, reason = expect_raises(RetrievalContractViolation, rc.validate,
                            [chunk("doc_a")], rule="min_results")
test("Too few results raises violation", ok, reason)

stale = [chunk("doc_a", age_days=5), chunk("doc_b", age_days=45)]
ok, reason = expect_raises(RetrievalContractViolation, rc.validate, stale, rule="max_age_days")
test("Stale chunk raises violation", ok, reason)

score = rc.check_drift(baseline, baseline)
test("Identical source sets: Jaccard=1.0", abs(score - 1.0) < 1e-9)

updated_drift = [chunk("doc_d"), chunk("doc_d"), chunk("doc_d")]
ok, reason = expect_raises(RetrievalContractViolation, rc.check_drift, baseline, updated_drift,
                            rule="min_jaccard_overlap")
test("Complete source swap raises drift violation", ok, reason)

# ===========================================================================
# EMBEDDING CONTRACT TESTS
# ===========================================================================
section("Embedding Contract")

from contracts.embedding import EmbeddingContract, EmbeddingContractViolation
import numpy as np

rng = np.random.default_rng(42)
def make_vecs(n, d, seed=42):
    v = np.random.default_rng(seed).normal(size=(n, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).tolist()

ec = EmbeddingContract(expected_dimensions=128, min_mean_cross_similarity=0.90,
                        min_neighbourhood_overlap=0.6)

vecs = make_vecs(20, 128)

ok, reason = expect_no_raise(ec.check_dimensions, vecs)
test("Correct dimensions pass", ok, reason)

wrong_dim = make_vecs(5, 256)
ok, reason = expect_raises(EmbeddingContractViolation, ec.check_dimensions, wrong_dim,
                            rule="expected_dimensions")
test("Wrong dimensions raise violation", ok, reason)

score = ec.check_distributional_stability(vecs, vecs)
test("Same vectors: distributional stability = 1.0", abs(score - 1.0) < 1e-6)

# Simulate model swap: generate entirely new random vectors (different model geometry)
model_v2_vecs = make_vecs(20, 128, seed=999)  # completely different orientation
ok, reason = expect_raises(EmbeddingContractViolation, ec.check_distributional_stability,
                            vecs, model_v2_vecs, rule="min_mean_cross_similarity")
test("Model swap (different random geometry) fails distributional check", ok, reason)

overlap = ec.check_neighbourhood_stability(vecs, vecs, k=5)
test("Same vectors: neighbourhood overlap = 1.0", abs(overlap - 1.0) < 1e-6)

ok, reason = expect_raises(EmbeddingContractViolation, ec.check_neighbourhood_stability,
                            vecs, model_v2_vecs, k=5, rule="min_neighbourhood_overlap")
test("Model swap destabilises neighbourhood rankings", ok, reason)

tiny = make_vecs(3, 128)
overlap = ec.check_neighbourhood_stability(tiny, tiny, k=5)
test("Too few vectors skips neighbourhood check gracefully", overlap == 1.0)

# ===========================================================================
# TOOL CONTRACT TESTS
# ===========================================================================
section("Tool Contract")

from contracts.tool import ToolContract, ToolContractViolation, ParameterConstraint

def make_flight_contract():
    c = ToolContract(
        tool_name="book_flight",
        parameter_constraints=[
            ParameterConstraint("origin", required=True, allowed_types=["str"], min_length=3, max_length=3),
            ParameterConstraint("destination", required=True, allowed_types=["str"], min_length=3, max_length=3),
            ParameterConstraint("depart_date", required=True, allowed_types=["str"]),
            ParameterConstraint("return_date", required=False, allowed_types=["str"]),
            ParameterConstraint("passengers", required=True, allowed_types=["int"], min_value=1, max_value=9),
            ParameterConstraint("cabin_class", required=True, allowed_values=["economy", "business", "first"]),
            ParameterConstraint("infant_passengers", required=False, allowed_types=["int"], min_value=0, max_value=4),
        ],
    )
    c.add_semantic_rule("return_after_departure", "return_date must be >= depart_date",
        lambda a: a.get("return_date", "9999-99-99") >= a["depart_date"])
    c.add_semantic_rule("no_infant_without_adult", "Infants require adult (passengers > infants)",
        lambda a: a.get("infant_passengers", 0) == 0 or a.get("passengers", 0) > a.get("infant_passengers", 0))
    c.add_semantic_rule("origin_ne_destination", "Origin and destination must differ",
        lambda a: a.get("origin","").upper() != a.get("destination","").upper())
    return c

fc = make_flight_contract()
valid_flight = {"origin": "LHR", "destination": "JFK", "depart_date": "2025-06-01",
                "return_date": "2025-06-15", "passengers": 2, "cabin_class": "economy"}

ok, reason = expect_no_raise(fc.validate, valid_flight)
test("Valid flight booking passes contract", ok, reason)

no_pax = {**valid_flight}; del no_pax["passengers"]
ok, reason = expect_raises(ToolContractViolation, fc.validate, no_pax, rule="required_parameter")
test("Missing required parameter raises violation", ok, reason)

wrong_type = {**valid_flight, "passengers": "two"}
ok, reason = expect_raises(ToolContractViolation, fc.validate, wrong_type, rule="parameter_type")
test("Wrong parameter type raises violation", ok, reason)

too_many = {**valid_flight, "passengers": 10}
ok, reason = expect_raises(ToolContractViolation, fc.validate, too_many, rule="parameter_max_value")
test("Passengers > max raises violation", ok, reason)

bad_class = {**valid_flight, "cabin_class": "premium_economy"}
ok, reason = expect_raises(ToolContractViolation, fc.validate, bad_class, rule="parameter_allowed_values")
test("Invalid cabin class raises violation", ok, reason)

# Semantic violations — structurally valid JSON, semantically impossible
return_early = {**valid_flight, "depart_date": "2025-06-15", "return_date": "2025-06-01"}
ok, reason = expect_raises(ToolContractViolation, fc.validate, return_early)
test("[SEMANTIC] Return before departure caught by contract", ok, reason)

same_airport = {**valid_flight, "destination": "LHR"}
ok, reason = expect_raises(ToolContractViolation, fc.validate, same_airport)
test("[SEMANTIC] Same origin/destination caught by contract", ok, reason)

all_infants = {**valid_flight, "passengers": 3, "infant_passengers": 3}
ok, reason = expect_raises(ToolContractViolation, fc.validate, all_infants)
test("[SEMANTIC] All-infant booking caught by contract", ok, reason)

# The key test: structurally perfect JSON, semantically impossible
impossible = {"origin": "LHR", "destination": "LHR", "depart_date": "2025-06-15",
              "return_date": "2025-06-01", "passengers": 3, "infant_passengers": 3,
              "cabin_class": "economy"}
ok, reason = expect_raises(ToolContractViolation, fc.validate, impossible)
test("[SEMANTIC] Structurally-valid-but-semantically-invalid call blocked", ok, reason)

# Forbidden combination test
msg_contract = ToolContract(tool_name="send_message",
    parameter_constraints=[ParameterConstraint("urgent"), ParameterConstraint("scheduled_for")],
    forbidden_arg_combinations=[["urgent", "scheduled_for"]])
ok, reason = expect_raises(ToolContractViolation, msg_contract.validate,
                            {"urgent": True, "scheduled_for": "2025-06-01"}, rule="forbidden_arg_combination")
test("Forbidden argument combination raises violation", ok, reason)

# Required group
lookup = ToolContract(tool_name="lookup_user", required_arg_groups=[["user_id", "email", "username"]])
ok, reason = expect_no_raise(lookup.validate, {"email": "a@b.com"})
test("Required group: at least one present passes", ok, reason)
ok, reason = expect_raises(ToolContractViolation, lookup.validate, {"full_name": "John"}, rule="required_arg_group")
test("Required group: none present raises violation", ok, reason)

# ===========================================================================
# SUMMARY
# ===========================================================================
total = results["passed"] + results["failed"] + results["skipped"]
print(f"\n{BOLD}{'='*60}")
print(f"Results: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped / {total} total{RESET}")

if failures:
    print(f"\n{BOLD}Failures:{RESET}")
    for name, reason in failures:
        print(f"  {FAIL} {name}")
        if reason:
            print(f"       {reason}")
    sys.exit(1)
else:
    print(f"\n{BOLD}\033[32mAll tests passed ✓{RESET}")
    sys.exit(0)
