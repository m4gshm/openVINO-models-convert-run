"""Tests of the command line option helpers."""
import unittest
from enum import Enum

from agent.common.cli import enum_value, enum_values


class Color(Enum):
    """A sample enum of the command line values."""
    RED = 'red'
    GREEN = 'green'


class EnumHelpersCase(unittest.TestCase):
    """Tests of the helpers that read the values of an enum."""

    def test_enum_value_is_the_raw_value(self):
        """The raw value of an enum member is returned."""
        self.assertEqual("red", enum_value(Color.RED))

    def test_enum_values_are_the_cli_choices(self):
        """The values of all members of the enum are returned."""
        self.assertEqual(["red", "green"], enum_values(Color))


if __name__ == '__main__':
    unittest.main()
