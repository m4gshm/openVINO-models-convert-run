"""Tests of the middleware checkpoint answer parsing."""
import unittest

from agent.openai.chat_api import ROLE_TOOL
from agent.openai.chat_completions_api import ChatCompletionMessageParam
from agent.openai.middleware_checkpoint import checkpoint_answer, is_middleware_checkpoint


class MiddlewareCheckpointCase(unittest.TestCase):
    """The answer must be read from the answer assignment, not from the echoed question."""

    def test_free_text_answer_of_an_echoed_question(self):
        """opencode echoes the question before the answer; the answer is the last assignment."""
        # the echoed question itself contains the word "interrupted" — a substring
        # search would mistake it for the "interrupt" answer (real session case)
        message = ChatCompletionMessageParam(role=ROLE_TOOL, tool_call_id="middleware_checkpoint_x",
                                             content='User has answered your questions: '
                                                     '"**WARNING: Cycled phrase detected**\n'
                                                     '```\nquestion body\n```\nThe generation kept '
                                                     'repeating itself and was interrupted after '
                                                     'retries.\n\n**What to do next?**"="ты зациклился". '
                                                     "You can now continue with the user's answers in mind.")
        self.assertTrue(is_middleware_checkpoint(message))
        self.assertEqual("ты зациклился", checkpoint_answer(message))

    def test_option_answer(self):
        """An option answer is the selected option label."""
        message = ChatCompletionMessageParam(role=ROLE_TOOL, tool_call_id="middleware_checkpoint_x",
                                             content='User has answered your questions: '
                                                     '"What to do next?"="interrupt". '
                                                     "You can now continue with the user's answers in mind.")
        self.assertEqual("interrupt", checkpoint_answer(message))

    def test_plain_content_without_assignment(self):
        """A reply without the answer assignment falls back to the whole content."""
        message = ChatCompletionMessageParam(role=ROLE_TOOL, tool_call_id="middleware_checkpoint_x",
                                             content="continue")
        self.assertEqual("continue", checkpoint_answer(message))


if __name__ == '__main__':
    unittest.main()
