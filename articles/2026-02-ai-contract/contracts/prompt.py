"""
contracts/prompt.py — Prompt Contract (pure Python, no pydantic)

The boundary between prompt and model output.
Treats prompts like APIs: explicit schemas, testable invariants.
"""

from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from typing import Any


class PromptContractViolation(Exception):
    def __init__(self, rule: str, detail: str, raw_response: str = ""):
        self.rule = rule
        self.detail = detail
        self.raw_response = raw_response
        super().__init__(f"PromptContract violation [{rule}]: {detail}")


@dataclass
class PromptContract:
    required_fields: list[str] = field(default_factory=list)
    forbidden_strings: list[str] = field(default_factory=list)
    allowed_confidence_values: list[str] | None = None
    must_cite_sources: bool = False
    max_tokens: int | None = None
    output_format: str = "json"

    def validate_response(self, raw_response: str) -> dict[str, Any]:
        if self.max_tokens is not None:
            if len(raw_response.split()) > self.max_tokens:
                raise PromptContractViolation("max_tokens",
                    f"Response ~{len(raw_response.split())} tokens exceeds {self.max_tokens}.", raw_response)

        lowered = raw_response.lower()
        for forbidden in self.forbidden_strings:
            if forbidden.lower() in lowered:
                raise PromptContractViolation("forbidden_string",
                    f"Contains forbidden string: '{forbidden}'.", raw_response)

        parsed: dict[str, Any] = {}
        if self.output_format == "json":
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE)
            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError as exc:
                raise PromptContractViolation("json_parse", f"Not valid JSON: {exc}", raw_response) from exc

        for f in self.required_fields:
            if f not in parsed:
                raise PromptContractViolation("required_field", f"Missing field '{f}'.", raw_response)

        if self.must_cite_sources:
            sources = parsed.get("sources", [])
            if not sources or not isinstance(sources, list):
                raise PromptContractViolation("must_cite_sources",
                    "Response must include non-empty 'sources' list.", raw_response)

        if self.allowed_confidence_values is not None and "confidence" in parsed:
            val = parsed["confidence"]
            if val not in self.allowed_confidence_values:
                raise PromptContractViolation("allowed_confidence_values",
                    f"confidence='{val}' not in {self.allowed_confidence_values}.", raw_response)

        return parsed

    def is_compliant(self, raw_response: str) -> tuple[bool, str]:
        try:
            self.validate_response(raw_response)
            return True, ""
        except PromptContractViolation as exc:
            return False, str(exc)