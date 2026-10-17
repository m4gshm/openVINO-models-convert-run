"""Base marker for Veai IDE tool call builders."""
from abc import ABC


class Tool(ABC):  # pylint: disable=too-few-public-methods  # marker base for IDE tool builders
    """Marker base class shared by all Veai IDE tool call builders."""
    # @staticmethod
    # @abstractmethod
    # def name(self) -> str:
    #     pass

    # @property
    # @abstractmethod
    # def parameters(self) -> set[str]:
    #     pass
