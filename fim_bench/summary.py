"""Summary table over all recorded FIM bench runs.

Reads the shared JSONL log written by ``python -m fim_bench`` (every run
appends one line per case) and renders a pivot table: one row per
(model, label) combination, one column per case, plus pass counts and times.

Usage:
  python -m fim_bench.summary
  python -m fim_bench.summary --results-log reports\\fim-results.jsonl --out reports\\summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path


def load_runs(results_log: Path) -> list[dict]:
    """Parse the JSONL log, skipping broken lines."""
    runs: list[dict] = []
    if not results_log.exists():
        return runs
    for line in results_log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return runs


def build_table(runs: list[dict]) -> tuple[list[str], list[dict]]:
    """Pivot the runs into (case_columns, rows).

    Each row: {model, label, runs, cells: {case: 'PASS 12.3s' | 'fail 4.5s' | 'ERR'}, passed, total}.
    """
    cases: list[str] = []
    seen_cases: set[str] = set()
    groups: "OrderedDict[tuple[str, str], list[dict]]" = OrderedDict()
    for run in runs:
        case = str(run.get("case", "?"))
        if case not in seen_cases:
            seen_cases.add(case)
            cases.append(case)
        key = (str(run.get("model", "?")), str(run.get("label", "default")))
        groups.setdefault(key, []).append(run)

    rows: list[dict] = []
    for (model, label), entries in groups.items():
        cells: dict[str, str] = {}
        passed = 0
        total = 0.0
        for entry in entries:
            case = str(entry.get("case", "?"))
            seconds = float(entry.get("seconds", 0.0))
            total += seconds
            if entry.get("error"):
                cells[case] = "ERR"
            elif entry.get("passed"):
                passed += 1
                cells[case] = f"PASS {seconds:.1f}s"
            else:
                cells[case] = f"fail {seconds:.1f}s"
        rows.append({
            "model": model,
            "label": label,
            "cases": cells,
            "passed": passed,
            "count": len(entries),
            "total": total,
        })
    return cases, rows


def format_markdown(cases: list[str], rows: list[dict]) -> str:
    """Render the pivot as a markdown table with column-aligned cells."""
    def case_cell(row: dict, case: str) -> str:
        return row["cases"].get(case, "-")

    table: list[list[str]] = []
    header = ["model", "label", *cases, "passed", "total s"]
    table.append(header)
    for row in rows:
        cells = [row["model"], row["label"]]
        cells += [case_cell(row, case) for case in cases]
        cells += [f"{row['passed']}/{row['count']}", f"{row['total']:.1f}"]
        table.append(cells)

    widths = [max(len(cells[col]) for cells in table) for col in range(len(header))]

    def fmt(cells: list[str]) -> str:
        padded = []
        for col, cell in enumerate(cells):
            is_right = header[col] in ("passed", "total s")
            padded.append(cell.rjust(widths[col]) if is_right else cell.ljust(widths[col]))
        return "| " + " | ".join(padded) + " |"

    lines = [fmt(header), "| " + " | ".join("-" * widths[col] for col in range(len(header))) + " |"]
    lines.extend(fmt(cells) for cells in table[1:])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fim_bench.summary",
                                     description="Summary table of all recorded FIM bench runs")
    parser.add_argument("--results-log", type=Path, default=Path("reports") / "fim-results.jsonl",
                        help="shared JSONL log of case results")
    parser.add_argument("--out", type=Path, default=None,
                        help="optional markdown file to write the table into")
    args = parser.parse_args(argv)

    def clickable(path: Path) -> str:
        return path.resolve().as_uri()

    runs = load_runs(args.results_log)
    if not runs:
        print(f"no runs recorded in {clickable(args.results_log)}")
        return 1
    cases, rows = build_table(runs)
    table = format_markdown(cases, rows)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
        print(f"[summary] written to {clickable(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
