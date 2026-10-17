"""Tests of the /v1/models remote modality reporting."""
import unittest

from openai.types.model import Model

from agent.openai.engine_rest_openai import remote_modalities


def new_model(**extra) -> Model:
    """Remote model entry as the OpenAI SDK parses it, with optional extension fields."""
    payload = {"id": "m", "object": "model", "created": 1, "owned_by": "o"} | extra
    return Model.model_validate(payload)


class RemoteModalitiesCase(unittest.TestCase):
    """Cases of remote_modalities() over openai SDK Model objects."""
    def test_extension_field_is_passed_through_sorted_and_deduplicated(self):
        """Extension lists are sorted and deduplicated."""
        model = new_model(supported_modalities=["text", "image", "text", "audio"])
        self.assertEqual(["audio", "image", "text"], remote_modalities(model))

    def test_absent_when_remote_is_silent(self):
        """No supported_modalities field means the value stays absent."""
        self.assertIsNone(remote_modalities(new_model()))

    def test_empty_declaration_is_unknown(self):
        """An empty modality list is reported as unknown."""
        self.assertIsNone(remote_modalities(new_model(supported_modalities=[])))

    def test_non_list_declaration_is_ignored(self):
        """A non-list modality declaration is ignored."""
        self.assertIsNone(remote_modalities(new_model(supported_modalities="image")))

    def test_blank_names_are_dropped(self):
        """Blank or non-string entries are dropped."""
        self.assertIsNone(remote_modalities(new_model(supported_modalities=["", None, 7])))

    def test_object_without_pydantic_extras(self):
        """Plain objects without model_extra yield None instead of raising."""
        class Bare:  # pylint: disable=too-few-public-methods  # value-object stub
            """Minimal object without pydantic extras support."""

        self.assertIsNone(remote_modalities(Bare()))


if __name__ == '__main__':
    unittest.main()
