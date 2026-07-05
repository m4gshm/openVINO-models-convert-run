import cProfile
import pstats
import re
import unittest
from importlib.resources import files

from agent.inference.loop_error import LoopError
from agent.inference.phrase import DUPLICATED_TOKENS_LIMIT, Phrase, add_token_to_current_line, \
    get_ranges_with_duplicates_started_by_token, merge_ranges

TEST_RESOURCES = "test_resources/phrase"


def visualize_duplicate_parts(current_line: list[str], duplicated_ranges: dict[int, int]) -> str:
    result = ""
    start = 0
    for dstart, damount in duplicated_ranges.items():
        if dstart > start:
            result += "-" * (dstart - start)
        dend = dstart + damount
        result += "".join(current_line[dstart:dend])
        start = dend
    return result


class PhraseTestCase(unittest.TestCase):
    def test_loop_tokens(self):
        repeated_string = "a" + ("b" * DUPLICATED_TOKENS_LIMIT)

        phrase = Phrase()
        with self.assertRaises(LoopError):
            for token in repeated_string:
                phrase.add_token(token)

    def test_loop_lines(self):
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "loop_messages.txt")
        loop_messages = loop_messages_file.read_text(encoding="utf-8")

        phrase = Phrase()
        with self.assertRaises(LoopError):
            tokens = re.split(r"([\n|\s])", loop_messages)
            for token in tokens:
                phrase.add_token(token)

        self.assertEqual('some normal output\n'
                         'first\n'
                         'second\n'
                         'first\n'
                         'second\n'
                         'first\n'
                         'second\n'
                         'first\n'
                         'second\n'
                         'first\n'
                         'second\n', phrase.full)

    def test_loop_in_one_line(self):
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line.txt")
        loop_messages = loop_messages_file.read_text(encoding="utf-8")

        current_line = list[str]()
        current_line_tokens = dict[str, list[int]]()

        for i, token in enumerate(loop_messages):
            add_token_to_current_line(token, current_line, current_line_tokens)

        result_ranges = {}
        for i, token in enumerate(loop_messages):
            ranges = get_ranges_with_duplicates_started_by_token(token, current_line_tokens, current_line)
            if ranges:
                parts = visualize_duplicate_parts(current_line, ranges)
                for k, v in ranges.items():
                    get = result_ranges.get(k, 0)
                    s = max(get, v)
                    result_ranges[k] = s

        result_ranges = merge_ranges(result_ranges)

        loop_part = result = visualize_duplicate_parts(current_line, result_ranges)
        self.assertEqual('----imoim-lloimport\\nimport\\nimportiiiii', loop_part)
        #----imoim-lloimport\nimport\nimportiiiii
        #ilheimoimalloimport\nimport\nimportiiiii

    def test_loop_in_one_line2(self):
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line2.txt").read_text(encoding="utf-8")
        loop_messages_expected_result = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line2_result.txt").read_text(encoding="utf-8")

        current_line = list[str]()
        current_line_tokens = dict[str, list[int]]()

        for i, token in enumerate(loop_messages):
            add_token_to_current_line(token, current_line, current_line_tokens)

        result_ranges = {}

        # profiler = cProfile.Profile()
        # profiler.enable()

        # for i, token in enumerate(loop_messages):
        token = 'i'
        ranges = get_ranges_with_duplicates_started_by_token(token, current_line_tokens, current_line)

        # profiler.disable()

        # Format and display the statistics
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(10)  # Print the top 10 bottlenecks

        if ranges:
            for k, v in ranges.items():
                get = result_ranges.get(k, 0)
                s = max(get, v)
                result_ranges[k] = s

        result = visualize_duplicate_parts(current_line, result_ranges)
        self.maxDiff = None
        self.assertEqual(loop_messages_expected_result, result)

        total_tokens = len(loop_messages)
        total_in_duplicates = 0
        max_part_in_duplicates = 0
        for start, amount in result_ranges.items():
            max_part_in_duplicates = max(max_part_in_duplicates, amount)
            total_in_duplicates += amount

        duplicates_rate = total_in_duplicates / total_tokens
        max_part_rate = max_part_in_duplicates / total_tokens
        
        pass




if __name__ == '__main__':
    unittest.main()
