"""Tests of the benchmark runner formatting."""

import unittest
from unittest import mock

from fim_bench.cases import Case
from fim_bench.models import ModelInfo, QWEN_STYLE
from fim_bench.runner import CaseResult, format_report, run_bench

INFO = ModelInfo("m", QWEN_STYLE)


class RunBenchCase(unittest.TestCase):
    """Tests of the case execution."""

    def test_run_bench_scores_the_key(self):
        cases = [Case("c1", "pre", "suf", "lambda")]
        with mock.patch("fim_bench.runner.CompletionClient") as client_cls:
            client_cls.return_value.fim.return_value = ("lambda x, y: x + y", 1.5)
            results = run_bench(INFO, 1, cases)
        self.assertEqual(1, len(results))
        self.assertTrue(results[0].passed)
        self.assertEqual("lambda", results[0].expected)

    def test_run_bench_survives_a_dead_server(self):
        cases = [Case("c1", "pre", "suf", "key")]
        with mock.patch("fim_bench.runner.CompletionClient") as client_cls:
            client_cls.return_value.fim.side_effect = ConnectionError("refused")
            results = run_bench(INFO, 1, cases)
        self.assertFalse(results[0].passed)
        self.assertIn("refused", results[0].error)
        self.assertEqual("key", results[0].expected)


class FormatReportCase(unittest.TestCase):
    """Tests of the report text."""

    def test_report_shows_expected_and_actual(self):
        results = [
            CaseResult("c1", True, 1.5, "lambda x, y: x + y", expected="lambda"),
            CaseResult("c2", False, 2.0, "\\n}\n}", expected="x * x"),
        ]
        report = format_report(INFO, results)
        self.assertIn("expected: lambda", report)
        self.assertIn("actual  : lambda x, y: x + y", report)
        self.assertIn("expected: x * x", report)
        self.assertIn("actual  : \\\\n}\\n}", report)
        self.assertIn("1/2 passed", report)

    def test_report_shows_error_with_expected(self):
        results = [CaseResult("c1", False, 0.0, "", expected="k", error="boom")]
        report = format_report(INFO, results)
        self.assertIn("[ERR ] c1: boom", report)
        self.assertIn("expected: k", report)


if __name__ == "__main__":
    unittest.main()
