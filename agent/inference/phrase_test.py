import cProfile
import pstats
import unittest
from importlib.resources import files
from typing import Any

from agent.inference.loop_error import LoopError
from agent.inference.phrase import DUPLICATED_TOKENS_LIMIT, Phrase, \
    get_duplicated_parts, \
    add_token_to_line, visualize_duplicated_positions, visualize_duplicated_parts

TEST_RESOURCES = "test_resources/phrase"


def merge(ranges: dict[int, int]) -> dict[Any, Any]:
    result_ranges = {}
    if ranges:
        for k, v in ranges.items():
            get = result_ranges.get(k, 0)
            s = max(get, v)
            result_ranges[k] = s
    return result_ranges


def merge_word(line: list[str], duplicated_ranges: dict[int, int], duplicated_words: dict[str, set[int]]) -> dict[
    int, int]:
    merged_ranges = duplicated_ranges.copy()
    merged_reversed_ranges = dict[int, int]()
    merged_token_positions = set[int]()
    for token_position, token in enumerate(line):
        # if token_position in merged_token_positions:
        #     continue
        word_start = token_position
        end = merged_ranges.get(word_start)
        if not end is None:
            word = "".join(line[word_start:end + 1])
            word_starts = duplicated_words.get(word, set())

            merged_word_on_position = dict[int, str]()
            merged_next_word_starts_on_position = dict[int, set[int]]()
            for i in range(1, len(word)):
                for word_start in word_starts:
                    word_end = word_start + len(word) - 1
                    next_word_start = word_start + i
                    is_merged = False
                    while next_word_start <= (word_end + 1):
                        next_word_end = merged_ranges.get(next_word_start)
                        if not next_word_end is None:
                            if next_word_end > word_end:
                                # merge
                                on_merge_word = "".join(line[next_word_start:next_word_end + 1])
                                # counts = duplicated_words.get(on_merge_word)
                                merged_word = "".join(line[word_start:next_word_end + 1])
                                is_merged = update_merged_by_max(merged_word_on_position, merged_word, word_start)
                                if is_merged:
                                    merged_next_word_starts_on_position.setdefault(word_start, set()).add(
                                        next_word_start)
                            else:
                                is_merged = True
                                # del internal word
                                del merged_ranges[next_word_start]
                                if next_word_end in merged_reversed_ranges:
                                    old_start = merged_reversed_ranges.get(next_word_end)
                                    if not old_start is None and old_start == next_word_start:
                                        del merged_reversed_ranges[next_word_end]
                                # todo del duplicated_word
                        next_word_start += 1
                    if not is_merged:
                        update_merged_by_max(merged_word_on_position, word, word_start)

            merged_words = dict[str, set[int]]()
            for word_start, word in merged_word_on_position.items():
                merged_words.setdefault(word, set()).add(word_start)

            for word, word_starts in merged_words.items():
                # if len(word_starts) > 1:
                for word_start in word_starts:
                    end = word_start + len(word) - 1
                    old_wold_end = merged_ranges.get(word_start)
                    if old_wold_end is not None and old_wold_end >= end:
                        pass
                    else:
                        prev_start = merged_reversed_ranges.get(end)
                        if prev_start is not None and prev_start <= word_start:
                            pass
                        else:
                            merged_reversed_ranges[end] = word_start
                            merged_ranges[word_start] = end
                            merged_words_starts = merged_next_word_starts_on_position.get(word_start)
                            if merged_words_starts:
                                for merged_word_start in merged_words_starts:
                                    old_end = merged_ranges.get(merged_word_start)
                                    if old_end is None:
                                        # error
                                        pass
                                    del merged_ranges[merged_word_start]
                                    old_start = merged_reversed_ranges.get(old_end)
                                    if not old_start is None and old_start == merged_word_start:
                                        del merged_reversed_ranges[old_end]

                                    new_start_of_word_tail = end + 1
                                    if new_start_of_word_tail <= old_end:
                                        exists_end = merged_ranges.get(new_start_of_word_tail)
                                        if exists_end is None or exists_end < old_end:
                                            merged_ranges[new_start_of_word_tail] = old_end
    return merged_ranges


def update_merged_by_max(merged_word_on_position: dict[int, str], merged_word: str, start: int) -> bool:
    merger_position_word = merged_word_on_position.get(start)
    if merger_position_word is None or len(merger_position_word) < len(merged_word):
        merged_word_on_position[start] = merged_word
        return True
    return False


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
        duplicated_reversed_ranges = dict[int, int]()
        duplicated_ranges = dict[int, int]()
        duplicated_positions = set[int]()
        duplicated_words = dict[str, set[int]]()

        profiler = cProfile.Profile()
        profiler.enable()

        for i, token in enumerate(loop_messages):
            add_token_to_line(token, line, line_tokens, duplicated_reversed_ranges, duplicated_ranges, duplicated_words,
                              duplicated_positions)

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        merged_ranges = merge_word(line, duplicated_ranges, duplicated_words)

        parts = get_duplicated_parts(line, duplicated_ranges)
        merged_parts = get_duplicated_parts(line, merged_ranges)

        loop_part1 = visualize_duplicated_parts(line, duplicated_ranges)
        loop_part2 = visualize_duplicated_parts(line, merged_ranges)
        loop_part3 = visualize_duplicated_positions(line, lambda i: i in duplicated_positions)

        self.assertEqual('----im-im----import\\nimport\\nimportiiii-', loop_part1)
        self.assertEqual(loop_part1, loop_part2)
        self.assertEqual(loop_part1, loop_part3)

        self.assertEqual({'ii': [35, 37], 'im': [7, 4, 29], 'import\\n': [13, 21], 'mport': [14, 22, 30]}, parts)
        self.assertEqual({'iiii': [35], 'im': [7, 4], 'import': [29], 'import\\nimport\\n': [13]}, merged_parts)

    def test_loop_in_one_line4_repeated_consonants(self):
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line4.txt")
        loop_messages = loop_messages_file.read_text(encoding="utf-8")

        line = list[str]()
        line_tokens = dict[str, list[int]]()
        duplicated_reversed_ranges = dict[int, int]()
        duplicated_ranges = dict[int, int]()
        duplicated_positions = set[int]()
        duplicated_words = dict[str, set[int]]()

        profiler = cProfile.Profile()
        profiler.enable()

        for i, token in enumerate(loop_messages):
            add_token_to_line(token, line, line_tokens, duplicated_reversed_ranges, duplicated_ranges, duplicated_words,
                              duplicated_positions)

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        merged_ranges = merge_word(line, duplicated_ranges, duplicated_words)

        parts = get_duplicated_parts(line, duplicated_ranges)
        merged_parts = get_duplicated_parts(line, merged_ranges)

        loop_part1 = visualize_duplicated_parts(line, duplicated_ranges)
        loop_part2 = visualize_duplicated_parts(line, merged_ranges)
        loop_part3 = visualize_duplicated_positions(line, lambda i: i in duplicated_positions)

        self.assertEqual(
            'ribute;At;PersistenceUnitXmlAttribute;PersistenceUnitXmlAttribute;PersistenceUnitXmlAttribute;',
            loop_part1)
        self.assertEqual(loop_part1, loop_part2)
        self.assertEqual(loop_part1, loop_part3)

        self.assertEqual({';PersistenceUnitXml': [9, 37, 65],
                          'At': [7],
                          'Att': [56, 28, 84],
                          'Per': [38, 10, 66],
                          'Un': [49, 21, 77],
                          'ce': [47, 19, 75],
                          'itX': [79, 23, 51],
                          'ribu': [31, 0, 59, 87],
                          'sis': [41, 13, 69],
                          'te;': [35, 4, 63, 91],
                          'ten': [16, 44, 72]}, parts)
        self.assertEqual({';PersistenceUnitXmlAttribute;': [37, 65, 9],
                          'At;PersistenceUnitXml': [7],
                          'ribute;': [0]}, merged_parts)

    def test_loop_in_one_line2(self):
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line2.txt").read_text(encoding="utf-8")
        loop_messages_expected_result = files(__package__).joinpath(TEST_RESOURCES,
                                                                    "loop_in_line2_result.txt").read_text(
            encoding="utf-8")
        loop_messages_expected_touched_result = files(__package__).joinpath(TEST_RESOURCES,
                                                                            "loop_in_line2_result2.txt").read_text(
            encoding="utf-8")

        line = list[str]()
        line_tokens = dict[str, list[int]]()
        duplicated_reversed_ranges = dict[int, int]()
        duplicated_ranges = dict[int, int]()
        duplicated_positions = set[int]()
        duplicated_words = dict[str, set[int]]()

        profiler = cProfile.Profile()
        profiler.enable()

        for i, token in enumerate(loop_messages):
            add_token_to_line(token, line, line_tokens, duplicated_reversed_ranges, duplicated_ranges, duplicated_words,
                              duplicated_positions)

        merged_ranges = merge_word(line, duplicated_ranges, duplicated_words)

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        parts = get_duplicated_parts(line, duplicated_ranges)
        merged_parts = get_duplicated_parts(line, merged_ranges)

        merged_ranges2 = merge_word(line, merged_ranges, {w: set(p) for w, p in merged_parts.items()})
        merged_parts2 = get_duplicated_parts(line, merged_ranges2)

        merged_ranges3 = merge_word(line, merged_ranges2, {w: set(p) for w, p in merged_parts2.items()})
        merged_parts3 = get_duplicated_parts(line, merged_ranges3)

        loop_part1 = visualize_duplicated_parts(line, duplicated_ranges)
        loop_part2 = visualize_duplicated_parts(line, merged_ranges)
        loop_part3 = visualize_duplicated_parts(line, merged_ranges2)
        loop_part4 = visualize_duplicated_positions(line, lambda i: i in duplicated_positions)

        self.maxDiff = None
        self.assertEqual(loop_messages_expected_result, loop_part1)
        self.assertEqual(loop_part1, loop_part2)
        self.assertEqual(loop_part1, loop_part3)
        self.assertEqual(loop_messages_expected_touched_result, loop_part4)

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
        duplicated_reversed_ranges = dict[int, int]()
        duplicated_ranges = dict[int, int]()
        duplicated_positions = set[int]()
        duplicated_words = dict[str, set[int]]()

        profiler = cProfile.Profile()
        profiler.enable()

        for i, token in enumerate(loop_messages):
            add_token_to_line(token, line, line_tokens, duplicated_reversed_ranges, duplicated_ranges, duplicated_words,
                              duplicated_positions)

        merged_ranges = merge_word(line, duplicated_ranges, duplicated_words)

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        parts = get_duplicated_parts(line, duplicated_ranges)
        merged_parts = get_duplicated_parts(line, merged_ranges)

        loop_part1 = visualize_duplicated_parts(line, duplicated_ranges)
        loop_part2 = visualize_duplicated_parts(line, merged_ranges)
        loop_part3 = visualize_duplicated_positions(line, lambda i: i in duplicated_positions)

        pass


if __name__ == '__main__':
    unittest.main()
