"""Tests of the Ollama-compatible API: request conversion and response shaping."""
import unittest
from typing import Any, cast

from agent.ollama.ollama_api import OllamaChatRequest, OllamaGenerateRequest, OllamaMessage, \
    OllamaOptions, OllamaShowRequest
from agent.ollama.ollama_convert import aggregate_ollama_chat, aggregate_ollama_generate, \
    chat_chunk_to_ollama, chat_chunks_to_ollama, done_reason_of, generate_chunk_to_ollama, \
    ollama_chat_to_request, ollama_generate_to_request, ollama_options_to_request_params, \
    tags_of_models
from agent.openai.chat_api import ROLE_ASSISTANT, new_chat_completion_chunk
from agent.openai.models_api import ModelObject


class OllamaOptionsCase(unittest.TestCase):
    """Tests of the mapping of the Ollama options onto the OpenAI request parameters."""

    def test_no_options_produce_no_parameters(self):
        """A request without options changes nothing."""
        self.assertEqual({}, ollama_options_to_request_params(None))

    def test_options_are_mapped(self):
        """The Ollama options reach the request under their OpenAI names."""
        params = ollama_options_to_request_params(OllamaOptions(temperature=0.3, top_p=0.8,
                                                                 num_predict=64, stop=["x"]))
        self.assertEqual(0.3, params["temperature"])
        self.assertEqual(0.8, params["top_p"])
        self.assertEqual(64, params["max_tokens"])
        self.assertEqual(["x"], params["stop"])


class OllamaChatConversionCase(unittest.TestCase):
    """Tests of the conversion of an Ollama chat request into an OpenAI one."""

    def test_messages_are_converted(self):
        """The Ollama messages become OpenAI chat messages."""
        body = OllamaChatRequest(model="m", messages=[OllamaMessage(role="user", content="hi")])
        request = ollama_chat_to_request(body)
        self.assertEqual("m", request.model)
        self.assertEqual(1, len(request.messages))
        self.assertEqual("user", request.messages[0].role)
        self.assertEqual("hi", request.messages[0].content)

    def test_images_become_content_parts(self):
        """The base64 images of a message become OpenAI image content parts."""
        body = OllamaChatRequest(model="m", messages=[
            OllamaMessage(role="user", content="look", images=["QUJD"])])
        request = ollama_chat_to_request(body)
        parts = cast(list[Any], request.messages[0].content)
        content = [part.model_dump() for part in parts]
        self.assertEqual("text", content[0]["type"])
        self.assertEqual("image_url", content[1]["type"])
        self.assertEqual("data:image/png;base64,QUJD", content[1]["image_url"]["url"])

    def test_options_are_applied_to_the_request(self):
        """The options of the request are applied to the converted one."""
        body = OllamaChatRequest(model="m", messages=[OllamaMessage(role="user", content="hi")],
                                 options=OllamaOptions(temperature=0.1, num_predict=8))
        request = ollama_chat_to_request(body)
        self.assertEqual(0.1, request.temperature)
        self.assertEqual(8, request.max_tokens)

    def test_stream_flag_is_kept(self):
        """The stream flag of the request is kept."""
        body = OllamaChatRequest(model="m", stream=False)
        self.assertFalse(ollama_chat_to_request(body).stream)


class OllamaGenerateConversionCase(unittest.TestCase):
    """Tests of the conversion of an Ollama generate request into an OpenAI one."""

    def test_prompt_becomes_a_user_message(self):
        """The raw prompt becomes the content of a user message."""
        body = OllamaGenerateRequest(model="m", prompt="hello")
        request = ollama_generate_to_request(body)
        self.assertEqual(1, len(request.messages))
        self.assertEqual("user", request.messages[0].role)
        self.assertEqual("hello", request.messages[0].content)

    def test_system_prompt_precedes_the_user_message(self):
        """The system prompt of the request becomes the first message."""
        body = OllamaGenerateRequest(model="m", prompt="hello", system="be brief")
        request = ollama_generate_to_request(body)
        self.assertEqual(2, len(request.messages))
        self.assertEqual("system", request.messages[0].role)
        self.assertEqual("be brief", request.messages[0].content)


class OllamaToolCase(unittest.TestCase):
    """Tests of the conversion of the Ollama tools."""

    def test_a_tool_becomes_a_function_tool(self):
        """The Ollama tool becomes an OpenAI function tool."""
        body = OllamaChatRequest(model="m", tools=[{"type": "function", "function": {
            "name": "read_file", "description": "read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}])

        request = ollama_chat_to_request(body)
        self.assertEqual(1, len(request.tools))
        self.assertEqual("read_file", request.tools[0].function.name)

    def test_a_loose_schema_gets_the_defaults(self):
        """A tool without a JSON schema type gets the default one."""
        body = OllamaChatRequest(model="m", tools=[{"type": "function", "function": {
            "name": "read_file"}}])
        request = ollama_chat_to_request(body)
        self.assertEqual("object", request.tools[0].function.parameters.type)
        self.assertEqual({}, request.tools[0].function.parameters.properties)


class OllamaResponseCase(unittest.TestCase):
    """Tests of the conversion of the OpenAI chunks into the Ollama response lines."""

    def test_chunk_becomes_a_message_line(self):
        """A content chunk becomes an Ollama message line."""
        chunk = new_chat_completion_chunk(role=ROLE_ASSISTANT, content="hi", model="m")
        line = chat_chunk_to_ollama(chunk, "m", "now")
        self.assertEqual("m", line["model"])
        self.assertEqual("assistant", line["message"]["role"])
        self.assertEqual("hi", line["message"]["content"])
        self.assertFalse(line["done"])

    def test_finish_reason_ends_the_stream(self):
        """A chunk with a finish reason marks the response as done."""
        chunk = new_chat_completion_chunk(role=ROLE_ASSISTANT, content="", finish_reason="stop")
        line = chat_chunk_to_ollama(chunk, "m", "now")
        self.assertTrue(line["done"])
        self.assertEqual("stop", line["done_reason"])

    def test_length_finish_reason_is_kept(self):
        """A length stop is reported as such."""
        self.assertEqual("length", done_reason_of("length"))
        self.assertEqual("stop", done_reason_of("tool_calls"))

    def test_lines_are_aggregated_into_one_response(self):
        """The streaming lines aggregate into a single response."""
        chunks = [new_chat_completion_chunk(role=ROLE_ASSISTANT, content="he", model="m"),
                  new_chat_completion_chunk(role=ROLE_ASSISTANT, content="llo", model="m"),
                  new_chat_completion_chunk(role=ROLE_ASSISTANT, content="", finish_reason="stop")]
        lines = list(chat_chunks_to_ollama(chunks, "m", "now"))
        response = aggregate_ollama_chat(lines)
        self.assertEqual("hello", response["message"]["content"])
        self.assertTrue(response["done"])
        self.assertEqual("stop", response["done_reason"])

    def test_generate_line_carries_the_prompt(self):
        """A generate line repeats the prompt of the request."""
        chunk = new_chat_completion_chunk(role=ROLE_ASSISTANT, content="hi", model="m")
        line = generate_chunk_to_ollama(chunk, "m", "now", "the prompt")
        self.assertEqual("the prompt", line["prompt"])
        self.assertEqual("hi", line["response"])

    def test_generate_lines_are_aggregated(self):
        """The generate lines aggregate into a single response."""
        chunks = [new_chat_completion_chunk(role=ROLE_ASSISTANT, content="a", model="m"),
                  new_chat_completion_chunk(role=ROLE_ASSISTANT, content="b", model="m"),
                  new_chat_completion_chunk(role=ROLE_ASSISTANT, content="", finish_reason="stop")]
        lines = [generate_chunk_to_ollama(chunk, "m", "now", "p") for chunk in chunks]
        response = aggregate_ollama_generate(lines)
        self.assertEqual("ab", response["response"])
        self.assertTrue(response["done"])


class OllamaTagsCase(unittest.TestCase):
    """Tests of the model list of the Ollama API."""

    def test_models_become_entries(self):
        """Every served model becomes an entry of the model list."""
        response = tags_of_models([ModelObject(id="model-a"), ModelObject(id="model-b")])
        self.assertEqual(2, len(response.models))
        self.assertEqual("model-a", response.models[0].name)
        self.assertEqual("model-a", response.models[0].model)
        self.assertEqual("model-b", response.models[1].name)


class OllamaShowCase(unittest.TestCase):
    """Tests of the model details of the Ollama API."""

    def test_show_request_keeps_the_model_name(self):
        """The show request keeps the name of the asked model."""
        self.assertEqual("m", OllamaShowRequest(model="m").model)


if __name__ == '__main__':
    unittest.main()
