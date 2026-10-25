"""Benchmark runner: executes the cases against one model and formats results."""

from __future__ import annotations

from dataclasses import dataclass

from fim_bench.cases import CASES, Case
from fim_bench.client import CompletionClient
from fim_bench.models import ModelInfo


@dataclass(frozen=True)
class CaseResult:
    """The outcome of a single FIM case."""

    case: str
    passed: bool
    seconds: float
    output: str
    expected: str = ""
    error: str = ""


def run_bench(info: ModelInfo, port: int, cases: list[Case] | None = None,
              warmup: bool = False) -> list[CaseResult]:
    """Run every case against a running server, never raises.

    With ``warmup`` a throwaway short request is sent first: the first
    inference after the model load uploads weights to the device and is
    far slower than the steady state, which would skew the timings.
    """
    client = CompletionClient(port, info)
    if warmup:
        try:
            client.fim("def add(a, b):\n    ", "\n", max_tokens=8)
            print("[bench] warmup done")
        except Exception as exc:  # noqa: BLE001 - warmup failure surfaces in the cases below
            print(f"[bench] warmup failed: {exc}")
    results: list[CaseResult] = []
    for case in cases or CASES:
        try:
            text, seconds = client.fim(case.prefix, case.suffix, max_tokens=case.max_tokens)
            results.append(CaseResult(case.name, case.key in text, seconds, text, expected=case.key))
        except Exception as exc:  # noqa: BLE001 - a broken server must not kill the run
            results.append(CaseResult(case.name, False, 0.0, "", expected=case.key, error=str(exc)))
    return results


def _inline(text: str) -> str:
    """Escape a completion for a single-line display."""
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def format_report(info: ModelInfo, results: list[CaseResult]) -> str:
    """Human-readable summary block of one model."""
    lines = [f"=== {info.name} ==="]
    for res in results:
        if res.error:
            lines.append(f"[ERR ] {res.case}: {res.error}")
            lines.append(f"       expected: {_inline(res.expected)}")
            continue
        mark = "PASS" if res.passed else "FAIL"
        lines.append(f"[{mark}] {res.case:14s} {res.seconds:6.1f}s")
        lines.append(f"       expected: {_inline(res.expected)}")
        lines.append(f"       actual  : {_inline(res.output)}")
    passed = sum(1 for r in results if r.passed)
    total = sum(r.seconds for r in results)
    lines.append(f"--- {info.name}: {passed}/{len(results)} passed, total {total:.1f}s")
    if info.note:
        lines.append(f"    note: {info.note}")
    return "\n".join(lines)
