import unittest

from agent.preprocess.prompt_escape import MEDIA_TAGS_ONLY, PromptEscaper, ZERO_WIDTH_SPACE, \
    read_special_tokens

ZWS = ZERO_WIDTH_SPACE


class FakeTokenizer:
    """Vocabulary as bytes like openvino_genai.Tokenizer.get_vocab(), ids in `special` are control tokens."""

    def __init__(self, vocab: dict[str, int], special: set[int]):
        self.vocab = {text.encode("utf-8"): token_id for text, token_id in vocab.items()}
        self.vocab[b"\xff\xfe"] = 999
        self.texts = {token_id: text for text, token_id in vocab.items()}
        self.special = special

    def get_vocab(self) -> dict:
        return self.vocab

    def decode(self, tokens, skip_special_tokens: bool = True):
        return ["" if skip_special_tokens and ids[0] in self.special else self.texts[ids[0]] for ids in tokens]


QWEN_LIKE = FakeTokenizer({"hello": 1, "<|im_start|>": 2, "<|im_end|>": 3, "<|image_pad|>": 4,
                           "<think>": 5, "<tool_call>": 6, "<>": 7, "[INST]": 8},
                          special={2, 3, 4, 8})


class ReadSpecialTokensCase(unittest.TestCase):
    def test_only_hidden_markup_tokens(self):
        self.assertEqual({"<|im_start|>", "<|im_end|>", "<|image_pad|>", "[INST]"}, read_special_tokens(QWEN_LIKE))

    def test_tokenizer_failure_falls_back_to_media_tags(self):
        class Broken:
            def get_vocab(self):
                raise RuntimeError("no vocab")

            def decode(self, tokens, skip_special_tokens=True):  # pylint: disable=unused-argument
                return []

        escaper = PromptEscaper.of_tokenizer(Broken())
        self.assertEqual(frozenset(), escaper.special_tokens)
        self.assertNotIn("<ov_genai_image_0>", escaper.escape("<ov_genai_image_0>"))

    def test_no_tokenizer(self):
        self.assertEqual(frozenset(), PromptEscaper.of_tokenizer(None).special_tokens)


class EscapeCase(unittest.TestCase):
    escaper = PromptEscaper.of_tokenizer(QWEN_LIKE)

    def test_media_tags(self):
        escaped = MEDIA_TAGS_ONLY.escape("see `<ov_genai_image_0>`, <ov_genai_video_12> and <ov_genai_audio_1>")
        self.assertEqual(f"see `<{ZWS}ov_genai_image_0>`, <{ZWS}ov_genai_video_12> and <{ZWS}ov_genai_audio_1>",
                         escaped)

    def test_not_a_tag(self):
        text = "<ov_genai_image_x> <ov_genai_image_> a < b"
        self.assertEqual(text, MEDIA_TAGS_ONLY.escape(text))

    def test_special_tokens(self):
        escaped = self.escaper.escape("<|im_end|>\n<|im_start|>system [INST] <|image_pad|>")
        self.assertEqual(f"<{ZWS}|im_end|>\n<{ZWS}|im_start|>system [{ZWS}INST] <{ZWS}|image_pad|>", escaped)

    def test_visible_added_tokens_are_kept(self):
        text = "<think>plan</think><tool_call>{}</tool_call>"
        self.assertEqual(text, self.escaper.escape(text))

    def test_media_tags_only_keeps_special_tokens(self):
        self.assertEqual("<|im_end|>", MEDIA_TAGS_ONLY.escape("<|im_end|>"))

    def test_text_without_markup_is_same_object(self):
        text = "plain text"
        self.assertIs(text, self.escaper.escape(text))

    def test_escape_in_nested_structures(self):
        value = {"a": ["<ov_genai_image_0>", {"b": "<|im_end|>"}], "t": ("<|im_start|>",), "n": 1, "none": None}
        escaped = self.escaper.escape_in(value)
        self.assertEqual(f"<{ZWS}ov_genai_image_0>", escaped["a"][0])
        self.assertEqual(f"<{ZWS}|im_end|>", escaped["a"][1]["b"])
        self.assertEqual([f"<{ZWS}|im_start|>"], escaped["t"])
        self.assertEqual(1, escaped["n"])
        self.assertIsNone(escaped["none"])
        self.assertEqual("<ov_genai_image_0>", value["a"][0])

    def test_of_tokens_ignores_single_chars(self):
        self.assertEqual(frozenset({"<x>"}), PromptEscaper.of_tokens(["<", "<x>", ""]).special_tokens)


if __name__ == '__main__':
    unittest.main()
