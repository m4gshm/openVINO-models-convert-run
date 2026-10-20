"""Helpers for the command line options of the server."""
from enum import Enum
from typing import Any, TypeVar

EnumT = TypeVar("EnumT", bound=Enum)


def enum_value(member: EnumT) -> Any:
    """Return the raw value of an enum member."""
    return member.value


def enum_values(enum_class: type[EnumT]) -> list[str]:
    """Return the CLI-choice values of an enum class."""
    return [member.value for member in enum_class]
