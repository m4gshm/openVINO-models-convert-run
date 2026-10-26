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

    def test_run_bench_accepts_an_alt_key(self):
        cases = [Case("c1", "pre", "suf", "a, b = b, a + b", alt_keys=("c = a + b",))]
        with mock.patch("fim_bench.runner.CompletionClient") as client_cls:
            client_cls.return_value.fim.return_value = ("\n        c = a + b\n        a = b\n        b = c\n", 0.8)
            results = run_bench(INFO, 1, cases)
        self.assertTrue(results[0].passed)
        self.assertEqual("a, b = b, a + b | c = a + b", results[0].expected)

    def test_expected_display_lists_alts_requirements_and_pattern(self):
        plain = Case("c1", "pre", "suf", "lambda").expected_display()
        self.assertEqual("lambda", plain)
        alts = Case("c2", "pre", "suf", "Set(", alt_keys=("indexOf(",)).expected_display()
        self.assertEqual("Set( | indexOf(", alts)
        conj = Case("c3", "pre", "suf", "x * x", all_keys=("int) int",)).expected_display()
        self.assertEqual("x * x (and: int) int)", conj)
        patterned = Case("c4", "pre", "suf", "x * x", pattern=r"(\w+) \* \1").expected_display()
        self.assertEqual(r"x * x (pattern: (\w+) \* \1)", patterned)

    def test_case_matches_rejects_wrong_text(self):
        case = Case("c1", "pre", "suf", "a, b = b, a + b", alt_keys=("c = a + b",))
        self.assertFalse(case.matches("while n > 0:\n        n -= 1"))
        self.assertTrue(case.matches("a, b = b, a + b"))
        self.assertTrue(case.matches("        c = a + b\n"))

    def test_case_matches_ignores_key_deep_in_the_tail(self):
        # a degenerate fill followed by invented boilerplate that happens
        # to mention the key must not pass
        case = Case("c1", "pre", "suf", "p1",
                    alt_keys=("s1",),
                    match_window=120)
        boilerplate = '*pointer.Of(&string{value: "test"}))' * 10 + "\n\tassert.Equal(t, p1, p2)"
        self.assertFalse(case.matches(boilerplate))
        self.assertTrue(case.matches(" p1)\n\tassert.Equal(t, *p1, *p2)"))

    def test_case_matches_requires_all_fragments(self):
        # a body fragment alone is not enough when the fill spans
        # signature + body
        case = Case("c1", "pre", "suf", "x * x", all_keys=("int) int",))
        self.assertFalse(case.matches("x * x"))                      # body only, no signature
        self.assertTrue(case.matches("x int) int {\n\treturn x * x"))  # full fill

    def test_case_matches_pattern_accepts_renamed_parameter(self):
        case = Case("c1", "pre", "suf", "x * x",
                    pattern=r"(\w+) int\) int \{[\s\S]{0,40}?return \1 \* \1")
        self.assertTrue(case.matches("n int) int { return n * n }"))
        self.assertTrue(case.matches("x int) int {\n\treturn x * x\n}\n\n"))
        self.assertFalse(case.matches("x int) int {\n\treturn y * y"))  # mismatched ids
        self.assertFalse(case.matches("x * x"))                          # no signature at all

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
