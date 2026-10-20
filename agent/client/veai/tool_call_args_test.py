"""Tests of the generic argument utilities of the Veai tool call repair."""
import unittest

from agent.client.user_context import OS, UserContext
from agent.client.veai.tool_call_args import as_bool_or_none, as_int_or_none, as_type_or_none, \
    clean_file_path, fix_windows_path, get_one_of, get_target_file


def windows_context() -> UserContext:
    """A user context of a Windows session."""
    context = UserContext()
    context.os_type = OS.Windows
    return context


class AsIntOrNoneCase(unittest.TestCase):
    """Tests of the integer coercion of the model arguments."""

    def test_int_passes(self):
        """An integer value is used as it is."""
        self.assertEqual((7, False), as_int_or_none(7, "value"))

    def test_numeric_string_is_cast(self):
        """A numeric string is cast to an integer."""
        self.assertEqual((7, False), as_int_or_none("7", "value"))

    def test_negative_string_is_cast(self):
        """A negative numeric string is cast to an integer."""
        self.assertEqual((-7, False), as_int_or_none("-7", "value"))

    def test_text_without_a_number_is_not_recovered(self):
        """A text without a number cannot be recovered, and the repair is reported."""
        self.assertEqual((None, True), as_int_or_none("abc", "value"))


class AsBoolOrNoneCase(unittest.TestCase):
    """Tests of the boolean coercion of the model arguments."""

    def test_bool_passes(self):
        """A boolean value is used as it is."""
        self.assertTrue(as_bool_or_none(True, "value"))
        self.assertFalse(as_bool_or_none(False, "value"))

    def test_string_is_mapped(self):
        """A string is mapped to the matching boolean."""
        self.assertTrue(as_bool_or_none("true", "value"))
        self.assertFalse(as_bool_or_none("false", "value"))

    def test_unrelated_type_is_its_own_truth_value(self):
        """A value of an unrelated type is cast by its truth value."""
        self.assertFalse(as_bool_or_none([], "value"))
        self.assertTrue(as_bool_or_none([1], "value"))


class AsTypeOrNoneCase(unittest.TestCase):
    """Tests of the generic cast of the model arguments."""

    def test_value_of_the_type_passes(self):
        """A value of the requested type is used as it is."""
        self.assertEqual(3, as_type_or_none(int, 3, "value"))

    def test_value_is_cast(self):
        """A value of another type is cast when it is possible."""
        self.assertEqual(3, as_type_or_none(int, "3", "value"))

    def test_bad_value_is_none(self):
        """A value that cannot be cast becomes None."""
        self.assertIsNone(as_type_or_none(int, "abc", "value"))


class GetOneOfCase(unittest.TestCase):
    """Tests of the lookup of an argument among its alternatives."""

    def test_first_set_field_wins(self):
        """The first field that is set is returned."""
        self.assertEqual("b", get_one_of({"a": None, "b": "b"}, ["a", "b"]))

    def test_missing_field_is_none(self):
        """None is returned when no field of the list is set."""
        self.assertIsNone(get_one_of({"c": "c"}, ["a", "b"]))


class GetTargetFileCase(unittest.TestCase):
    """Tests of the resolution of the target file of a tool call."""

    def test_primary_field_is_used(self):
        """The canonical field of the target file is used as it is."""
        target_file, invalid = get_target_file({"target_file": "a.txt"}, None)
        self.assertEqual("a.txt", target_file)
        self.assertFalse(invalid)

    def test_alias_field_is_used(self):
        """An alias of the target file is accepted."""
        target_file, invalid = get_target_file({"file_path": "a.txt"}, None)
        self.assertEqual("a.txt", target_file)
        self.assertTrue(invalid)

    def test_missing_target_file_is_invalid(self):
        """A missing target file is reported as invalid."""
        target_file, invalid = get_target_file({}, None)
        self.assertIsNone(target_file)
        self.assertTrue(invalid)


class CleanFilePathCase(unittest.TestCase):
    """Tests of the cleanup of a file path."""

    def test_braces_are_dropped(self):
        """The stray braces of a file path are dropped."""
        self.assertEqual("a.txt", clean_file_path("a.txt}"))

    def test_quotes_are_dropped(self):
        """The stray quotes of a file path are dropped."""
        self.assertEqual("a.txt", clean_file_path('"a.txt"'))

    def test_escaped_quote_is_kept(self):
        """An escaped quote is part of the name and stays."""
        self.assertEqual('a\\".txt', clean_file_path('a\\".txt'))

    def test_empty_path_is_kept(self):
        """An empty path stays empty."""
        self.assertEqual("", clean_file_path(""))


class FixWindowsPathCase(unittest.TestCase):
    """Tests of the normalization of the paths of a Windows session."""

    def test_posix_path_is_normalized(self):
        """A POSIX path of a Windows session loses its leading slash."""
        path, fixed = fix_windows_path("/src/a.txt", windows_context())
        self.assertTrue(fixed)
        self.assertFalse(path.startswith("/"))

    def test_posix_path_gets_a_drive_letter(self):
        """A POSIX path of a Windows session with a drive letter keeps it."""
        path, fixed = fix_windows_path("/c/src/a.txt", windows_context())
        self.assertTrue(fixed)
        self.assertTrue(path.startswith("c:"))

    def test_windows_path_is_kept(self):
        """A path that already has a drive letter is kept as it is."""
        path, fixed = fix_windows_path("C:/src/a.txt", windows_context())
        self.assertFalse(fixed)
        self.assertEqual("C:/src/a.txt", path)

    def test_path_of_another_system_is_kept(self):
        """A path of a session that is not Windows is kept as it is."""
        path, fixed = fix_windows_path("/src/a.txt", UserContext())
        self.assertFalse(fixed)
        self.assertEqual("/src/a.txt", path)

    def test_missing_path_is_kept(self):
        """A missing path stays missing."""
        path, fixed = fix_windows_path(None, windows_context())
        self.assertIsNone(path)
        self.assertFalse(fixed)


if __name__ == '__main__':
    unittest.main()
