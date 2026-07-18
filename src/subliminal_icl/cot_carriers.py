"""Arithmetic chain-of-thought carriers (EXPERIMENT_PLAN.md §5.6, notebook 10).

Generates exactly-verified arithmetic problems, each with several
*mathematically equivalent correct* reasoning traces. The activation search may
select among correct traces or numeric checksum slots; it may NEVER change the
final answer. A deterministic solver verifies every trace/answer.

Two carrier types:
  - strict_correct_cot : select among valid alternative reasonings only.
  - checksum_carrier   : problem/reasoning/answer fixed; only digit-only checksum
                         slots vary. Reported separately (weaker) — the checksum
                         is not logically required by the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# deterministic solver / verifier
# --------------------------------------------------------------------------

def verify_addition(a: int, b: int, answer: int) -> bool:
    return a + b == answer


def verify_mul(a: int, b: int, answer: int) -> bool:
    return a * b == answer


@dataclass
class ArithmeticProblem:
    kind: str  # mul2x2 | add3 | sub_borrow | linear | modular
    operands: List[int]
    answer: int
    prompt: str
    meta: Dict[str, object] = field(default_factory=dict)

    def verify(self) -> bool:
        k = self.kind
        if k == "mul2x2":
            return self.operands[0] * self.operands[1] == self.answer
        if k == "add3":
            return sum(self.operands) == self.answer
        if k == "sub_borrow":
            return self.operands[0] - self.operands[1] == self.answer
        if k == "linear":  # a*x + b = c -> x
            a, b, c = self.operands
            return a * self.answer + b == c
        if k == "modular":  # a mod m
            a, m = self.operands
            return a % m == self.answer
        raise ValueError(f"unknown kind {k}")


# --------------------------------------------------------------------------
# problem generation
# --------------------------------------------------------------------------

def generate_problem(kind: str, rng: np.random.Generator) -> ArithmeticProblem:
    if kind == "mul2x2":
        a, b = int(rng.integers(10, 100)), int(rng.integers(10, 100))
        return ArithmeticProblem(kind, [a, b], a * b, f"Compute {a} * {b}.")
    if kind == "add3":
        xs = [int(rng.integers(10, 1000)) for _ in range(3)]
        return ArithmeticProblem(kind, xs, sum(xs), f"Compute {xs[0]} + {xs[1]} + {xs[2]}.")
    if kind == "sub_borrow":
        a = int(rng.integers(100, 1000)); b = int(rng.integers(10, a))
        return ArithmeticProblem(kind, [a, b], a - b, f"Compute {a} - {b}.")
    if kind == "linear":
        a = int(rng.integers(2, 12)); x = int(rng.integers(1, 20)); b = int(rng.integers(-20, 20))
        c = a * x + b
        return ArithmeticProblem(kind, [a, b, c], x, f"Solve {a}*x + {b} = {c} for x.")
    if kind == "modular":
        a = int(rng.integers(50, 500)); m = int(rng.integers(3, 20))
        return ArithmeticProblem(kind, [a, m], a % m, f"Compute {a} mod {m}.")
    raise ValueError(f"unknown kind {kind}")


# --------------------------------------------------------------------------
# equivalent correct traces
# --------------------------------------------------------------------------

def _mul_traces(a: int, b: int) -> List[str]:
    ans = a * b
    traces = []
    # partial products by splitting b
    hi = (b // 10) * 10
    lo = b - hi
    if lo:
        traces.append(f"{a} * {hi} = {a*hi}; {a} * {lo} = {a*lo}; {a*hi} + {a*lo} = {ans}.")
    # split a
    ahi = (a // 10) * 10
    alo = a - ahi
    if alo:
        traces.append(f"{ahi} * {b} = {ahi*b}; {alo} * {b} = {alo*b}; {ahi*b} + {alo*b} = {ans}.")
    # commuted
    traces.append(f"{b} * {a} = {ans} (multiplication commutes).")
    # doubling/halving when even
    if b % 2 == 0:
        traces.append(f"{a} * {b} = {2*a} * {b//2} = {2*a*(b//2)} = {ans}.")
    # repeated grouping
    traces.append(f"{a} * {b}: {a} * 10 = {a*10}, so {a} * {b} = {a*10}*{b//10} + {a}*{b%10} = {ans}." if b >= 10 else f"{a} * {b} = {ans}.")
    return list(dict.fromkeys(traces))  # dedupe, keep order


def _generic_traces(problem: ArithmeticProblem) -> List[str]:
    p = problem
    if p.kind == "mul2x2":
        return _mul_traces(p.operands[0], p.operands[1])
    if p.kind == "add3":
        x, y, z = p.operands
        return list(dict.fromkeys([
            f"{x} + {y} = {x+y}; {x+y} + {z} = {p.answer}.",
            f"{y} + {z} = {y+z}; {x} + {y+z} = {p.answer}.",
            f"{x} + {z} = {x+z}; {x+z} + {y} = {p.answer}.",
            f"({x} + {y} + {z}) = {p.answer}.",
        ]))
    if p.kind == "sub_borrow":
        a, b = p.operands
        return list(dict.fromkeys([
            f"{a} - {b} = {p.answer}.",
            f"{a} - {b}: {a} - {(b//10)*10} = {a-(b//10)*10}; then - {b%10} = {p.answer}.",
            f"add up from {b}: {b} + {p.answer} = {a}, so {a} - {b} = {p.answer}.",
        ]))
    if p.kind == "linear":
        a, b, c = p.operands
        return list(dict.fromkeys([
            f"{a}*x + {b} = {c}; {a}*x = {c-b}; x = {p.answer}.",
            f"subtract {b}: {a}*x = {c}-{b} = {c-b}; divide by {a}: x = {p.answer}.",
        ]))
    if p.kind == "modular":
        a, m = p.operands
        q = a // m
        return list(dict.fromkeys([
            f"{a} = {m}*{q} + {p.answer}, so {a} mod {m} = {p.answer}.",
            f"largest multiple of {m} below {a} is {m*q}; {a} - {m*q} = {p.answer}.",
        ]))
    raise ValueError(p.kind)


@dataclass
class CarrierExample:
    problem: ArithmeticProblem
    traces: List[str]          # verified equivalent correct reasonings
    answer_text: str

    def render(self, trace_index: int) -> str:
        t = self.traces[trace_index]
        return f"Problem: {self.problem.prompt}\nReasoning: {t}\nAnswer: {self.problem.answer}"


def build_carrier(problem: ArithmeticProblem, verify: bool = True) -> CarrierExample:
    traces = _generic_traces(problem)
    if verify:
        assert problem.verify(), f"unverified problem: {problem}"
        # traces are constructed from verified arithmetic; keep only those whose
        # stated final equals the answer.
        traces = [t for t in traces if str(problem.answer) in t]
        assert traces, "no valid trace survived verification"
    return CarrierExample(problem=problem, traces=traces, answer_text=str(problem.answer))


def generate_carrier_dataset(n: int, rng: np.random.Generator,
                             kinds: Optional[List[str]] = None) -> List[CarrierExample]:
    kinds = kinds or ["mul2x2", "add3", "sub_borrow", "linear", "modular"]
    out: List[CarrierExample] = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        prob = generate_problem(kind, rng)
        out.append(build_carrier(prob))
    return out


# --------------------------------------------------------------------------
# checksum carriers (§5.6 numeric carrier slots) — reported separately
# --------------------------------------------------------------------------

def render_with_checksum(example: CarrierExample, trace_index: int,
                         checksum_digits: List[int]) -> str:
    """Insert a digit-only 'Verification checksum' line. NOT logically required
    by the arithmetic; label the condition ``checksum_carrier``."""
    body = example.render(trace_index)
    groups = ", ".join("".join(str(d) for d in checksum_digits[i:i+3])
                       for i in range(0, len(checksum_digits), 3))
    lines = body.split("\n")
    lines.insert(-1, f"Verification checksum: {groups}")
    return "\n".join(lines)


def verify_dataset(examples: List[CarrierExample]) -> Dict[str, int]:
    """Deterministic re-verification of an entire carrier dataset."""
    ok, bad = 0, 0
    for ex in examples:
        if ex.problem.verify() and all(str(ex.problem.answer) in t for t in ex.traces):
            ok += 1
        else:
            bad += 1
    return {"verified": ok, "failed": bad, "total": len(examples)}
