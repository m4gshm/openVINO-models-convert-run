"""Tests of the FIM case definitions."""

import unittest

from fim_bench.cases import CASES


class CasePrefixesCase(unittest.TestCase):
    """Tests of the case prompt shapes."""

    def test_every_prefix_is_at_least_256_chars(self):
        """Realistic cases: a short prefix does not exercise long-context prefill."""
        for case in CASES:
            self.assertGreaterEqual(len(case.prefix), 256, case.name)

    def test_prefixes_end_mid_construction(self):
        """The cursor sits where code is genuinely incomplete."""
        for case in CASES:
            self.assertNotEqual(case.prefix.rstrip(), case.prefix.rstrip(";:{(\n"), case.name)


class CaseMatchingCase(unittest.TestCase):
    """Sanity checks of the acceptance rules on realistic completions."""

    def _match(self, name: str, completion: str) -> bool:
        case = next(c for c in CASES if c.name == name)
        return case.matches(completion)

    def test_fib_loop_accepts_swap_and_temp(self):
        self.assertTrue(self._match("py-fib-loop", "        a, b = b, a + b"))
        self.assertTrue(self._match("py-fib-loop", "        t = a + b\n        a = b\n        b = t"))
        self.assertFalse(self._match("py-fib-loop", "        a = b"))

    def test_go_square_accepts_renamed_param(self):
        self.assertTrue(self._match("go-square", "x int) int {\n\treturn x * x"))
        self.assertTrue(self._match("go-square", "n int) int {\n\treturn n * n"))
        self.assertFalse(self._match("go-square", "x int) int {\n\treturn x + x"))

    def test_json_dump(self):
        self.assertTrue(self._match("py-json-dump", '        json.dump(data, f, ensure_ascii=False)'))
        self.assertFalse(self._match("py-json-dump", "        f.write(data)"))

    def test_js_dedupe_accepts_known_idioms(self):
        self.assertTrue(self._match("js-dedupe", "[...new Set(arr)]"))
        self.assertTrue(self._match("js-dedupe", "arr.filter((v, i) => arr.indexOf(v) === i)"))
        self.assertFalse(self._match("js-dedupe", "arr.sort()"))

    def test_sum_reduce_accepts_lambda_and_operator_add(self):
        self.assertTrue(self._match("py-sum-reduce", "lambda acc, x: acc + x, nums"))
        self.assertTrue(self._match("py-sum-reduce", "add, nums, 0"))
        self.assertFalse(self._match("py-sum-reduce", "mul, nums, 0"))

    def test_go_long_ctx(self):
        self.assertTrue(self._match("go-long-ctx", "s1)"))
        self.assertTrue(self._match("go-long-ctx", "&s1)"))


if __name__ == "__main__":
    unittest.main()
