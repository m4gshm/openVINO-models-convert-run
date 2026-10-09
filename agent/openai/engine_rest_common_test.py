import base64
import unittest

from agent.multimodal.image_decode_test import new_png, rgb_pixels
from agent.multimodal.modality import Modalities, Modality
from agent.multimodal.request_media import prepare_messages
from agent.openai.chat_completions_api import ChatCompletionMessageParam
from agent.openai.engine_rest_common import ControllerConfig, new_chat_history


def new_config(**kwargs) -> ControllerConfig:
    return ControllerConfig(max_prompt_len=4096, model_architectures=set(), **kwargs)


class ControllerConfigCase(unittest.TestCase):
    def test_text_only_by_default(self):
        config = new_config()
        self.assertEqual(["text"], config.modalities_of().names())

    def test_text_is_always_reported(self):
        config = new_config(modalities={"image"})
        self.assertTrue(config.modalities_of().supports(Modality.TEXT))
        self.assertTrue(config.modalities_of().supports(Modality.IMAGE))

    def test_media_limits_come_from_config(self):
        config = new_config(max_media_bytes=1024, max_media_items=3,
                            allow_local_media_files=True, media_http_timeout=1.5)
        limits = config.media_limits()
        self.assertEqual(1024, limits.max_bytes)
        self.assertEqual(3, limits.max_items)
        self.assertTrue(limits.allow_local_files)
        self.assertEqual(1.5, limits.http_timeout)


class NewChatHistoryCase(unittest.TestCase):
    def test_request_models_are_accepted(self):
        messages = [ChatCompletionMessageParam(role="user", content="hi")]
        stored = new_chat_history(messages).get_messages()
        self.assertEqual("user", stored[0]["role"])
        self.assertEqual("hi", stored[0]["content"])

    def test_tools_are_registered(self):
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        history = new_chat_history([{"role": "user", "content": "hi"}], tools)
        self.assertEqual(1, len(history.get_tools()))

    def test_media_tags_reach_the_history(self):
        png = "data:image/png;base64," + base64.b64encode(new_png(rgb_pixels(2, 2))).decode()
        messages = [{"role": "user", "content": [{"type": "text", "text": "describe "},
                                                 {"type": "image_url", "image_url": {"url": png}}]}]
        prepared = prepare_messages(messages, Modalities.of(Modality.IMAGE))
        history = new_chat_history(prepared.messages)
        stored = history.get_messages()
        self.assertEqual("describe <ov_genai_image_0>", stored[0]["content"])
        self.assertEqual(1, len(prepared.media.images))


if __name__ == '__main__':
    unittest.main()
