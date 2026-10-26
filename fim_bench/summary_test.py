"""Tests of the summary pivot table."""

import json
import tempfile
import unittest
from pathlib import Path

from fim_bench.summary import build_table, format_markdown, load_runs, main


def _entry(model, label, case, passed, seconds):
    return {"model": model, "label": label, "case": case, "passed": passed, "seconds": seconds,
            "expected": "k", "actual": "k"}


class SummaryCase(unittest.TestCase):
    """Tests of loading, pivoting and rendering."""

    def test_load_runs_skips_broken_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(
                json.dumps(_entry("m", "STATIC", "c1", True, 1.0)) + "\n" + "garbage\n",
                encoding="utf-8")
            runs = load_runs(path)
        self.assertEqual(1, len(runs))

    def test_build_table_pivots_by_model_and_label(self):
        runs = [
            _entry("A", "STATIC", "c1", True, 1.0),
            _entry("A", "STATIC", "c2", False, 2.0),
            _entry("A", "PYRAMID", "c1", True, 0.5),
            _entry("B", "STATIC", "c1", False, 3.0),
        ]
        cases, rows = build_table(runs)
        self.assertEqual(["c1", "c2"], cases)
        self.assertEqual(3, len(rows))
        static_a = next(r for r in rows if r["model"] == "A" and r["label"] == "STATIC")
        self.assertEqual("PASS 1.0s", static_a["cases"]["c1"])
        self.assertEqual("fail 2.0s", static_a["cases"]["c2"])
        self.assertEqual(1, static_a["passed"])
        self.assertEqual(3.0, static_a["total"])

    def test_markdown_renders_columns(self):
        cases, rows = build_table([
            _entry("A", "STATIC", "c1", True, 1.0),
            _entry("B-model-longer-name", "STATIC", "c1", True, 12.34),
        ])
        table = format_markdown(cases, rows)
        lines = table.splitlines()
        self.assertEqual(4, len(lines))
        # all rows share the same width: cells are padded to the column width
        self.assertEqual(1, len({len(ln) for ln in lines}))
        # times are right-aligned within the shared width
        self.assertTrue(lines[2].rstrip().endswith("1.0 |"))
        self.assertTrue(lines[3].rstrip().endswith("12.3 |"))
        # the longer model name defines the column width for both rows
        self.assertIn("| B-model-longer-name |", lines[3])
        self.assertIn("| A                   |", lines[2])

    def test_main_reports_empty_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(["--results-log", str(Path(tmp) / "missing.jsonl")])
        self.assertEqual(1, rc)


if __name__ == "__main__":
    unittest.main()
