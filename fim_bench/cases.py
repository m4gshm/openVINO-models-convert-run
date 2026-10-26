"""The FIM tasks of the benchmark.

Each case is a (prefix, suffix) pair around an invisible cursor plus a key
fragment a good completion should contain. The keys are intentionally simple
substrings: the bench is a regression smoke, not an exact-match grader.

To add a case: append a :class:`Case` to ``CASES``. Long-context cases can
reuse :func:`go_padding` (or define their own padding helpers) to inflate the
prompt to the sizes the real IDE produces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    """One fill-in-the-middle task.

    ``key`` is the primary fragment a good completion should contain;
    ``alt_keys`` lists equally valid alternatives (e.g. the same loop
    body written with an explicit temp variable). ``all_keys`` lists
    fragments that must *all* be present too (AND over the OR of the
    keys), e.g. both the signature and the body of a function fill.
    ``pattern`` is an optional regular expression accepted instead of
    the keys — useful when the completion is free to rename identifiers
    (a backreference can then require the body to reuse the parameter).
    Keys are matched only
    within the first ``match_window`` characters of the completion: a
    key found deep in the tail is boilerplate the model invented after
    the actual fill, not the fill itself.
    """

    name: str
    prefix: str
    suffix: str
    key: str
    alt_keys: tuple[str, ...] = ()
    all_keys: tuple[str, ...] = ()
    pattern: str = ""
    match_window: int = 120
    max_tokens: int = 96

    def matches(self, text: str) -> bool:
        """True if the completion head contains the key or an accepted alternative."""
        head = text[: self.match_window]
        if self.pattern:
            return re.search(self.pattern, head) is not None
        key_ok = self.key in head or any(alt in head for alt in self.alt_keys)
        return key_ok and all(frag in head for frag in self.all_keys)

    def expected_display(self) -> str:
        """All accepted answers for the report: the key, alternatives and required fragments."""
        parts = [self.key, *self.alt_keys]
        text = " | ".join(parts)
        if self.all_keys:
            text += f" (and: {' & '.join(self.all_keys)})"
        if self.pattern:
            text += f" (pattern: {self.pattern})"
        return text


GO_CTX = """package test

import (
\t"testing"

\t"github.com/stretchr/testify/assert"
\t"github.com/m4gshm/gollections/slice"
)

"""


def go_padding(repeats: int = 8) -> str:
    """A block of realistic Go tests used to inflate long-context cases."""
    block = """func Test_CloneN(t *testing.T) {
\ttype entity struct{ val string }
\tvar (
\t\tfirst  = entity{"first"}
\t\tsecond = entity{"second"}
\t\tthird  = entity{"third"}
\t\tentities = []*entity{&first, &second, &third}
\t\tc        = clone.Of(entities)
\t)
\tassert.Equal(t, entities, c)
}

"""
    return block * repeats


CASES: list[Case] = [
    Case(
        "py-fib-loop",
        '"""Numeric helpers of the report service."""\n\n'
        "from typing import Iterable\n\n\n"
        "def fib_memo(n: int, _cache: dict[int, int] | None = None) -> int:\n"
        '    """Return the n-th Fibonacci number, memoized version."""\n'
        "    if _cache is None:\n"
        "        _cache = {0: 0, 1: 1}\n"
        "    if n not in _cache:\n"
        "        _cache[n] = fib_memo(n - 1, _cache) + fib_memo(n - 2, _cache)\n"
        "    return _cache[n]\n\n\n"
        "def bounded(nums: Iterable[int], limit: int) -> list[int]:\n"
        "    return [x for x in nums if x < limit]\n\n\n"
        "def fibonacci(n: int) -> int:\n"
 '    """Return the n-th Fibonacci number (iterative, O(1) memory)."""\n'
        "    if n <= 1:\n        return n\n"
        "    a, b = 0, 1\n    for _ in range(n - 1):",
        "    return a\n\nprint(fibonacci(10))  # 55",
        "a, b = b, a + b",
        # the same swap written with an explicit temp variable is equally correct
        alt_keys=("c = a + b", "tmp = a + b", "t = a + b"),
    ),
    Case(
        "go-square",
        GO_CTX
        + "func Test_SquareTable(t *testing.T) {\n"
        "\tfor _, tc := range []struct{ in, want int }{{1, 1}, {2, 4}, {5, 25}, {-2, 4}} {\n"
        "\t\tassert.Equal(t, tc.want, square(tc.in))\n\t}\n}\n\n"
        "func Test_SquareString(t *testing.T) {\n"
        "\tassert.Equal(t, \"\", squareString(\"\"))\n}\n\n"
        "func square(",
        "\n\nfunc Test_Negative(t *testing.T) {\n\tassert.Equal(t, 4, square(-2))\n}",
        "x * x",
        # the fill spans params + signature close + body; the parameter
        # name is the model's choice, but the body must square *its* parameter
        pattern=r"(\w+) int\) int \{[\s\S]{0,40}?return \1 \* \1",
    ),
    Case(
        "py-json-dump",
        '"""Persistence helpers of the ingest pipeline."""\n\n'
        "import json\nimport os\nfrom pathlib import Path\n\n"
        "DEFAULT_DIR = Path(os.environ.get(\"INGEST_DIR\", \"data\"))\n\n\n"
        "def dump_bytes(name: str, payload: bytes) -> Path:\n"
        "    path = DEFAULT_DIR / name\n"
        "    path.write_bytes(payload)\n"
        "    return path\n\n\n"
        "def save(path, data):\n"
        '    """Serialize ``data`` as UTF-8 JSON next to the raw payload."""\n'
        '    with open(path, "w", encoding="utf-8") as f:\n',
        "        f\n\n\ndef load(path):\n    with open(path, encoding=\"utf-8\") as f:\n        return json.load(f)",
        "json.dump",
    ),
    Case(
        "js-dedupe",
        "// Utility helpers for the report filters.\n\n"
        "const byCountDesc = (a, b) => b.count - a.count;\n\n"
        "function countEntries(entries) {\n"
        "    return entries.reduce((acc, e) => acc.set(e, (acc.get(e) ?? 0) + 1), new Map());\n"
        "}\n\n"
        "/**\n"
        " * Removes duplicate entries from an array of primitives.\n"
        " * Keeps the first occurrence, preserves the original order.\n"
        " */\n"
        "function dedupe(arr) {\n    return ",
        ";\n}\n\nconsole.log(dedupe([1, 2, 2, 3]));  // [1, 2, 3]",
        "new Set(",
        # any known dedupe idiom: Set/Map based, or a uniqueness check
        # inside a filter/indexOf/includes/some/findIndex chain
        pattern=r"new Set\(|new Map\(|\.indexOf\(|\.includes\(|\.some\(|\.findIndex\(",
    ),
    Case(
        "py-sum-reduce",
        '"""Aggregation helpers for the metrics dashboard."""\n\n'
        "from functools import reduce\nfrom operator import add, mul\n\n\n"
        "def product(nums) -> float:\n"
        '    """Product of the numbers, 1 for an empty iterable."""\n'
        "    return reduce(mul, nums, 1)\n\n\n"
        "def spread(nums) -> float:\n"
        "    vals = list(nums)\n"
        "    return max(vals) - min(vals) if vals else 0.0\n\n\n"
        "def total(nums):\n"
        '    """Sum of the numbers, 0 for an empty iterable."""\n'
        "    return reduce(",
        ")\n\nassert total([1, 2, 3]) == 6",
        "lambda",
        # add is an equally idiomatic reducer (it is imported right above)
        alt_keys=("add",),
    ),
    Case(
        "go-long-ctx",
        GO_CTX + go_padding() + 'func Test_PointerClone(t *testing.T) {\n\ts1 := "test"\n\tp1 := &s1\n\tp2 := clone.Ptr(',
        ")\n\tassert.Equal(t, s1, *p2)\n}",
        "p1",
        # clone.Ptr(s1) is the idiomatic fill and satisfies the suffix assertion too
        alt_keys=("s1",),
    ),
]
