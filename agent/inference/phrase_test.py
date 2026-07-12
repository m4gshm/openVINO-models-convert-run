import cProfile
import pstats
import unittest
from importlib.resources import files
from typing import Any

from agent.inference.loop_error import LoopError
from agent.inference.phrase import DUPLICATED_TOKENS_LIMIT, Phrase, \
    get_duplicated_parts, \
    add_token_to_line, visualize_reversed_ranges, visualize_duplicated_positions, reverse

TEST_RESOURCES = "test_resources/phrase"


def merge(ranges: dict[int, int]) -> dict[Any, Any]:
    result_ranges = {}
    if ranges:
        for k, v in ranges.items():
            get = result_ranges.get(k, 0)
            s = max(get, v)
            result_ranges[k] = s
    return result_ranges


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
            for token in loop_messages:
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

        line = list[str]()
        line_tokens = dict[str, list[int]]()
        reversed_ranges = dict[int, int]()
        duplicated_positions = set[int]()
        duplicated_words = dict[str, set[int]]()

        profiler = cProfile.Profile()
        profiler.enable()

        for i, token in enumerate(loop_messages):
            add_token_to_line(token, line, line_tokens, reversed_ranges, duplicated_words, duplicated_positions)

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        parts = get_duplicated_parts(line, reverse(reversed_ranges))

        loop_part1 = visualize_reversed_ranges(line, reversed_ranges)
        loop_part2 = visualize_duplicated_positions(line, lambda i: i in duplicated_positions)

        self.assertEqual('----im-im----import\\nimport\\nimportiiii-', loop_part1)
        self.assertEqual('----im-im----import\\nimport\\nimportiiii-', loop_part2)

    def test_loop_in_one_line2(self):
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line2.txt").read_text(encoding="utf-8")
        loop_messages_expected_result = files(__package__).joinpath(TEST_RESOURCES,
                                                                    "loop_in_line2_result.txt").read_text(
            encoding="utf-8")

        line = list[str]()
        line_tokens = dict[str, list[int]]()
        reversed_ranges = dict[int, int]()
        duplicated_positions = set[int]()
        duplicated_words = dict[str, set[int]]()

        profiler = cProfile.Profile()
        profiler.enable()

        for i, token in enumerate(loop_messages):
            add_token_to_line(token, line, line_tokens, reversed_ranges, duplicated_words, duplicated_positions)

        # for word in list(duplicated_words.keys()):
        #     ends = duplicated_words[word]
        #     if len(ends) == 1:
        #         print("remove "+word)
        #         del duplicated_words[word]
        #         for end in ends:
        #             del reversed_ranges[end]



        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        loop_part1 = visualize_reversed_ranges(line, reversed_ranges)
        loop_part2 = visualize_duplicated_positions(line, lambda i: i in duplicated_positions)

        self.maxDiff = None
        self.assertEqual(loop_part2, loop_part1)
        self.assertEqual(loop_messages_expected_result, loop_part1)

        # total_tokens = len(loop_messages)
        # total_in_duplicates = 0
        # max_part_in_duplicates = 0
        # for start, amount in result_ranges.items():
        #     max_part_in_duplicates = max(max_part_in_duplicates, amount)
        #     total_in_duplicates += amount
        #
        # duplicates_rate = total_in_duplicates / total_tokens
        # max_part_rate = max_part_in_duplicates / total_tokens

        pass

    def test_loop_in_one_line3(self):
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line3_success_case.txt").read_text(
            encoding="utf-8")

        line = list[str]()
        line_tokens = dict[str, list[int]]()
        reversed_ranges = dict[int, int]()
        duplicated_positions = set[int]()

        profiler = cProfile.Profile()
        profiler.enable()

        for i, token in enumerate(loop_messages):
            add_token_to_line(token, line, line_tokens, reversed_ranges, duplicated_positions)

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        loop_part1 = visualize_reversed_ranges(line, reversed_ranges)
        loop_part2 = visualize_duplicated_positions(line, lambda i: i in duplicated_positions)

        pass


if __name__ == '__main__':
    unittest.main()
