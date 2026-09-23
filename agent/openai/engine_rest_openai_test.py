"""
Tests for OpenAiController - OpenAI-compatible controller without OpenVINO dependencies.
"""
import threading
import time
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock
from datetime import timedelta

import pytest

from agent.openai.engine_rest_openai import OpenAiController
from agent.openai import GenerateOpts
from agent.openai.engine_rest_common import ControllerConfig
from agent.inference.token_handler import TokenHandlerConfig
from agent.parser import Parser


class MockTokenizer:
    """Mock tokenizer for testing."""
    def encode(self, text):
        mock_result = MagicMock()
        mock_result.size = len(text.split())
        return mock_result


class MockGenerationConfig:
    """Mock generation config for testing."""
    def __init__(self):
        self.max_length = 4096
        self.temperature = 0.7
        self.max_new_tokens = 1024
        self.top_p = 0.9
        self.frequency_penalty = 0.0
        self.stop_strings = {"stop"}


class TestOpenAiControllerInit:
    """Test OpenAiController initialization."""

    def test_init_basic(self):
        """Test basic initialization with minimal parameters."""
        config = ControllerConfig(
            model_name="test-model",
            max_prompt_len=4096,
            model_architectures={"llama"}
        )
        parser = MagicMock(spec=Parser)
        handler_config = TokenHandlerConfig()
        stop_signal = threading.Event()
        generate_opts = GenerateOpts()
        
        controller = OpenAiController(
            config=config,
            parser=parser,
            api_key="test-key",
            base_url="https://api.test.com/v1",
            handler_config=handler_config,
            stop_signal=stop_signal,
            generate_opts=generate_opts
        )
        
        assert controller.api_key == "test-key"
        assert controller.base_url == "https://api.test.com/v1"
        assert controller.model_name_override == "test-model"
        assert controller.client is not None

    def test_init_with_model_override(self):
        """Test initialization with model name override."""
        config = ControllerConfig(
            model_name="original-model",
            max_prompt_len=4096,
            model_architectures={"llama"}
        )
        parser = MagicMock(spec=Parser)
        handler_config = TokenHandlerConfig()
        stop_signal = threading.Event()
        generate_opts = GenerateOpts()
        
        controller = OpenAiController(
            config=config,
            parser=parser,
            api_key="test-key",
            base_url="https://api.test.com/v1",
            handler_config=handler_config,
            stop_signal=stop_signal,
            generate_opts=generate_opts,
            model_name_override="override-model"
        )
        
        assert controller.model_name_override == "override-model"


class TestOpenAiControllerChunkGenerator:
    """Test OpenAiController chunk_generator method."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock AsyncOpenAI client."""
        client = MagicMock(spec=['chat'])
        chat = MagicMock()
        completions = MagicMock()
        create = MagicMock()
        
        chat.completions = completions
        completions.create = create
        
        return client

    @pytest.fixture
    def controller_setup(self):
        """Setup controller for testing."""
        config = ControllerConfig(
            model_name="test-model",
            max_prompt_len=4096,
            model_architectures={"llama"}
        )
        parser = MagicMock(spec=Parser)
        handler_config = TokenHandlerConfig()
        stop_signal = threading.Event()
        generate_opts = GenerateOpts()
        
        controller = OpenAiController(
            config=config,
            parser=parser,
            api_key="test-key",
            base_url="https://api.test.com/v1",
            handler_config=handler_config,
            stop_signal=stop_signal,
            generate_opts=generate_opts
        )
        
        return controller

    def test_chunk_generator_success(self, controller_setup, mock_client):
        """Test successful chunk generation."""
        # Mock the client initialization
        with patch.object(controller_setup, 'client', mock_client):
            # Mock stream response
            mock_chunk_1 = MagicMock()
            mock_chunk_1.choices = [MagicMock()]
            mock_chunk_1.choices[0].delta = MagicMock(content="Hello", role="assistant")
            mock_chunk_1.choices[0].finish_reason = None
            
            mock_chunk_2 = MagicMock()
            mock_chunk_2.choices = [MagicMock()]
            mock_chunk_2.choices[0].delta = MagicMock(content=" world", role="assistant")
            mock_chunk_2.choices[0].finish_reason = "stop"
            
            mock_stream = [mock_chunk_1, mock_chunk_2]
            mock_client.chat.completions.create.return_value = iter(mock_stream)
            
            generation_config = MockGenerationConfig()
            token_handler = MagicMock()
            
            chunks = list(controller_setup.chunk_generator("Test prompt", generation_config, token_handler))
            
            assert len(chunks) > 0
            mock_client.chat.completions.create.assert_called_once()

    def test_chunk_generator_with_error(self, controller_setup, mock_client):
        """Test error handling in chunk generation."""
        with patch.object(controller_setup, 'client', mock_client):
            # Mock API error
            from openai import APIConnectionError
            mock_client.chat.completions.create.side_effect = APIConnectionError(message="Connection error", request=None)
            
            generation_config = MockGenerationConfig()
            token_handler = MagicMock()
            
            chunks = list(controller_setup.chunk_generator("Test prompt", generation_config, token_handler))
            
            # Should get error response
            assert len(chunks) > 0
            assert any("Connection error" in str(chunk) for chunk in chunks)


class TestOpenAiControllerSlots:
    """Test OpenAiController slots endpoint."""

    @pytest.fixture
    def controller_setup(self):
        """Setup controller for testing."""
        config = ControllerConfig(
            model_name="test-model",
            max_prompt_len=4096,
            model_architectures={"llama"}
        )
        parser = MagicMock(spec=Parser)
        handler_config = TokenHandlerConfig()
        stop_signal = threading.Event()
        generate_opts = GenerateOpts()
        
        controller = OpenAiController(
            config=config,
            parser=parser,
            api_key="test-key",
            base_url="https://api.test.com/v1",
            handler_config=handler_config,
            stop_signal=stop_signal,
            generate_opts=generate_opts
        )
        
        return controller

    def test_slots_returns_json_response(self, controller_setup):
        """Test that slots endpoint returns JSON response."""
        # Simulate the async call
        response = controller_setup.slots_sync()
        
        assert response is not None
        content = response.body.decode('utf-8')
        assert '"slots"' in content
        assert '"active_slots"' in content


class TestOpenAiControllerShutdown:
    """Test OpenAiController shutdown."""

    @pytest.fixture
    def controller_setup(self):
        """Setup controller for testing."""
        config = ControllerConfig(
            model_name="test-model",
            max_prompt_len=4096,
            model_architectures={"llama"}
        )
        parser = MagicMock(spec=Parser)
        handler_config = TokenHandlerConfig()
        stop_signal = threading.Event()
        generate_opts = GenerateOpts()
        
        controller = OpenAiController(
            config=config,
            parser=parser,
            api_key="test-key",
            base_url="https://api.test.com/v1",
            handler_config=handler_config,
            stop_signal=stop_signal,
            generate_opts=generate_opts
        )
        
        return controller

    def test_shutdown(self, controller_setup):
        """Test controller shutdown."""
        controller_setup.shutdown()
        assert controller_setup.closed.is_set()