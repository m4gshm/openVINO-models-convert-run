"""The FIM tasks of the benchmark.

Each case is a (prefix, suffix) pair around an invisible cursor plus a key
fragment a good completion should contain. The keys are intentionally simple
substrings: the bench is a regression smoke, not an exact-match grader.

To add a case: append a :class:`Case` to ``CASES``. Long-context cases can
reuse :func:`go_padding` (or define their own padding helpers) to inflate the
prompt to the sizes the real IDE produces.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    """One fill-in-the-middle task."""

    name: str
    prefix: str
    suffix: str
    key: str
    max_tokens: int = 96


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
        'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'
        "    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):",
        "    return a\n\nprint(fibonacci(10))  # 55",
        "a, b = b, a + b",
    ),
    Case(
        "go-square",
        GO_CTX + "func Test_Square(t *testing.T) {\n\tassert.Equal(t, 25, square(5))\n}\n\nfunc square(",
        "\n\nfunc Test_Negative(t *testing.T) {\n\tassert.Equal(t, 4, square(-2))\n}",
        "x * x",
    ),
    Case(
        "py-json-dump",
        'import json\n\ndef save(path, data):\n    with open(path, "w", encoding="utf-8") as f:\n',
        "        f\n\n\ndef load(path):\n    with open(path, encoding=\"utf-8\") as f:\n        return json.load(f)",
        "json.dump",
    ),
    Case(
        "js-dedupe",
        "function dedupe(arr) {\n    return ",
        ";\n}\n\nconsole.log(dedupe([1, 2, 2, 3]));  // [1, 2, 3]",
        "Set(",
    ),
    Case(
        "py-sum-reduce",
        "from functools import reduce\n\ndef total(nums):\n    return reduce(",
        ")\n\nassert total([1, 2, 3]) == 6",
        "lambda",
    ),
    Case(
        "go-long-ctx",
        GO_CTX + go_padding() + 'func Test_PointerClone(t *testing.T) {\n\ts1 := "test"\n\tp1 := &s1\n\tp2 := clone.Ptr(',
        ")\n\tassert.Equal(t, s1, *p2)\n}",
        "p1",
    ),
]
