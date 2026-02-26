"""

The boundary between generation and tool execution.
Validates both schema (structural) AND semantic invariants.
Structurally valid JSON can still violate real-world constraints.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolContractViolation(Exception):
    def __init__(self, rule: str, detail: str, call: dict[str, Any] | None = None):
        self.rule = rule
        self.detail = detail
        self.call = call
        super().__init__(f"ToolContract violation [{rule}]: {detail}")


@dataclass
class ParameterConstraint:
    name: str
    required: bool = False
    allowed_types: list[str] = field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None
    min_length: int | None = None
    max_length: int | None = None
    description: str = ""


@dataclass
class ToolContract:
    tool_name: str
    parameter_constraints: list[ParameterConstraint] = field(default_factory=list)
    forbidden_arg_combinations: list[list[str]] = field(default_factory=list)
    required_arg_groups: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        self._semantic_rules: list[tuple[str, str, Callable]] = []

    def add_semantic_rule(self, name: str, description: str,
                          predicate: Callable[[dict[str, Any]], bool]) -> "ToolContract":
        self._semantic_rules.append((name, description, predicate))
        return self

    def validate(self, args: dict[str, Any]) -> None:
        TYPE_MAP = {"str": str, "int": int, "float": (int, float),
                    "bool": bool, "list": list, "dict": dict}

        for c in self.parameter_constraints:
            if c.required and c.name not in args:
                raise ToolContractViolation("required_parameter",
                    f"Required parameter '{c.name}' is missing.", args)

        for c in self.parameter_constraints:
            if c.name not in args:
                continue
            v = args[c.name]
            if c.allowed_types:
                types = tuple(TYPE_MAP[t] for t in c.allowed_types if t in TYPE_MAP)
                if types and not isinstance(v, types):
                    raise ToolContractViolation("parameter_type",
                        f"'{c.name}' has type {type(v).__name__}, expected {c.allowed_types}.", args)
            if c.min_value is not None and isinstance(v, (int, float)):
                if v < c.min_value:
                    raise ToolContractViolation("parameter_min_value",
                        f"'{c.name}'={v} below minimum {c.min_value}.", args)
            if c.max_value is not None and isinstance(v, (int, float)):
                if v > c.max_value:
                    raise ToolContractViolation("parameter_max_value",
                        f"'{c.name}'={v} exceeds maximum {c.max_value}.", args)
            if c.allowed_values is not None and v not in c.allowed_values:
                raise ToolContractViolation("parameter_allowed_values",
                    f"'{c.name}'={v!r} not in {c.allowed_values}.", args)
            if hasattr(v, "__len__"):
                if c.min_length is not None and len(v) < c.min_length:
                    raise ToolContractViolation("parameter_min_length",
                        f"'{c.name}' length {len(v)} below minimum {c.min_length}.", args)
                if c.max_length is not None and len(v) > c.max_length:
                    raise ToolContractViolation("parameter_max_length",
                        f"'{c.name}' length {len(v)} exceeds maximum {c.max_length}.", args)

        for combo in self.forbidden_arg_combinations:
            if all(k in args for k in combo):
                raise ToolContractViolation("forbidden_arg_combination",
                    f"Arguments {combo} must not all be present simultaneously.", args)

        for group in self.required_arg_groups:
            if not any(k in args for k in group):
                raise ToolContractViolation("required_arg_group",
                    f"At least one of {group} must be present.", args)

        for name, description, predicate in self._semantic_rules:
            if not predicate(args):
                raise ToolContractViolation(f"semantic_rule:{name}", description, args)
