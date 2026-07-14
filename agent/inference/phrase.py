import logging
from typing import Any, Callable

from agent.inference.loop_error import LoopError

IGNORE_DUPLICATED_PARTS_SIZE = 10

INIT_STEP = 1

DUPLICATED_LINES_THRESHOLD = 5
DUPLICATED_TOKEN_THRESHOLD = 10

DUPLICATED_TOKENS_LIMIT = 100
DUPLICATED_LINES_RATE_LIMIT = 0.5
DUPLICATED_LINES_LIMIT = 50

log = logging.getLogger(__name__)


def merge_intersect_ranges(duplicate_ranges: dict[int, int]) -> tuple[dict[int, int], dict[int, int], list[int]]:
    bitset = 0
    last = 0
    for start, end in duplicate_ranges.items():
        last = max(last, end)
        for position in range(start, end + 1):
            bitset |= 1 << (position)

    merged_ranges = dict[int, int]()
    calculated_merged_part_size = dict[int, int]()
    last_positions = list[int]()
    start: int | None = None
    for position in range(last + 1):
        is_in_range = bool(bitset & (1 << position))
        if is_in_range:
            if start is None:
                start = position
        elif start:
            end = position - 1
            merged_ranges[start] = end
            calculated_merged_part_size[end] = end - start + 1
            last_positions.append(end)
            start = None
    return merged_ranges, calculated_merged_part_size, last_positions


def filter_ranges(duplicate_ranges: dict[int, int], filter_leq: int | None = None) -> dict[int, int]:
    if filter_leq:
        for start in list(duplicate_ranges.keys()):
            amount = duplicate_ranges[start]
            if amount <= filter_leq:
                del duplicate_ranges[start]

    return duplicate_ranges


# def visualize_reversed_ranges(line: list[str], reversed_ranges: dict[int, int]) -> str:
#     duplicated_ranges = reverse(reversed_ranges)
#     return visualize_duplicated_parts(line, duplicated_ranges)
#
#
# def reverse(reversed_ranges: dict[int, int]) -> dict[int, int]:
#     duplicated_ranges = dict[int, int]()
#     for end, start in reversed_ranges.items():
#         exists_end = duplicated_ranges.get(start)
#         if exists_end is None:
#             duplicated_ranges[start] = end
#         else:
#             duplicated_ranges[start] = max(end, exists_end)
#     return duplicated_ranges


def visualize_duplicated_parts(line: list[str], duplicated_ranges: dict[int, int]) -> str:
    only_duplicates_line: list[str] = ["-"] * len(line)
    for start, end in duplicated_ranges.items():
        i = start
        while i <= end:
            only_duplicates_line[i] = line[i]
            i += 1

    result = "".join(only_duplicates_line)
    return result


def visualize_duplicated_positions(line: list[str], is_in_duplicated: Callable[[int], bool]) -> str:
    only_duplicates_line: list[str] = ["-"] * len(line)
    for i, token in enumerate(line):
        if is_in_duplicated(i):
            only_duplicates_line[i] = token

    result = "".join(only_duplicates_line)
    return result


def get_duplicated_parts(line: list[str], duplicated_ranges: dict[int, int]) -> dict[str, list[int]]:
    parts = dict[str, list[int]]()
    for start, end in duplicated_ranges.items():
        part = "".join(line[start:end + 1])
        parts.setdefault(part, list[int]()).append(start)
    return parts


def register_to_check_duplicates(token: str, line: list[str],
                                 line_tokens: dict[str, list[int]],
                                 duplicated_reversed_ranges: dict[int, int],
                                 duplicated_ranges: dict[int, int],
                                 duplicated_words: dict[str, set[int]],
                                 duplicated_positions: set[int]):
    token_positions = line_tokens[token]
    if len(token_positions) <= 1:
        return

    exists_phrases = dict[str, set[int]]()
    token_new_positions = list[int]()
    duplicated_phrases = dict[str, set[int]]()
    prev_phrase_ref = dict[int, set[int]]()
    touched_positions = set[int]()
    for i, token_position in enumerate(token_positions):
        prev_token_position = token_position - 1
        next_token_position = token_position + 1

        if token_position <= 0:
            continue
        if prev_token_position < 0:
            continue
        if prev_token_position in touched_positions:
            # intersection
            continue
        # next_step_phrase_start = duplicated_reversed_ranges.get(next_token_position)
        # if not next_step_phrase_start is None:
        #     next_phrase_len = next_token_position - next_step_phrase_start + 1
        #     if next_phrase_len > 1:
        #         # already inside another word, skip
        #         continue
        next_step_phrase_end = duplicated_ranges.get(prev_token_position)
        # if not next_step_phrase_end is None:
        #     # already inside another word, skip
        #     continue
        next_step_phrase_end = duplicated_ranges.get(token_position)
        # if not next_step_phrase_end is None:
        #     # already inside another word, skip
        #     continue

        exists_phrase_start = duplicated_reversed_ranges.get(token_position)
        if_new_token = True
        if not exists_phrase_start is None:
            exists_phrase = get_word(line, exists_phrase_start, token_position)
            exists_phrases.setdefault(exists_phrase, set()).add(token_position)
            if_new_token = False
        elif i == len(token_positions) - 1:
            # compare the newest with exists phrases
            for exist_phrase, exist_phrase_positions in exists_phrases.items():
                exist_phrase_size = len(exist_phrase)
                possible_phrase = get_word(line, (token_position - (exist_phrase_size - 1)), token_position)
                if possible_phrase == exist_phrase:
                    for exist_phrase_position in exist_phrase_positions:
                        if_new_token = False
                        duplicated_phrase_positions = duplicated_phrases.setdefault(possible_phrase, set())
                        duplicated_phrase_positions.add(exist_phrase_position)
                        duplicated_phrase_positions.add(token_position)

        if if_new_token:
            token_new_positions.append(token_position)

    current_token_phrase = token

    any_expected_phrase_lengths = set[int]()
    for i, token_position in enumerate(token_new_positions):
        prev_token_position = token_position - 1

        prev_phrase_start = duplicated_reversed_ranges.get(prev_token_position)
        if not prev_phrase_start is None:
            prev_tokens_phrase = get_word(line, prev_phrase_start, prev_token_position)
            any_expected_phrase_lengths.add(len(prev_tokens_phrase))

    for i, token_position in enumerate(token_new_positions):
        prev_token_position = token_position - 1

        phrase_expected_default = get_word(line, prev_token_position, token_position)
        duplicated_phrases.setdefault(phrase_expected_default, set()).add(token_position)
        expected_phrase_by_prev_step_position = {phrase_expected_default: None}

        prev_phrase_start = duplicated_reversed_ranges.get(prev_token_position)
        if not prev_phrase_start is None:
            prev_tokens_phrase = get_word(line, prev_phrase_start, prev_token_position)
            prev_tokens_phrase_len = len(prev_tokens_phrase)
            expected_token_phrase = prev_tokens_phrase + current_token_phrase
            expected_phrase_by_prev_step_position[expected_token_phrase] = prev_token_position
            duplicated_phrases.setdefault(expected_token_phrase, set()).add(token_position)
            prev_phrase_ref.setdefault(token_position, set()).add(prev_token_position)
            for any_expected_phrase_len in any_expected_phrase_lengths:
                if prev_tokens_phrase_len != any_expected_phrase_len:
                    prev_phrase_start = prev_token_position - (any_expected_phrase_len - 1)
                    expected_token_phrase = get_word(line, prev_phrase_start, prev_token_position)
                    expected_phrase_by_prev_step_position[expected_token_phrase] = prev_token_position
                    duplicated_phrases.setdefault(expected_token_phrase, set()).add(token_position)

    concatenated_phrases = dict[str, set[int]]()
    concatenated_prev_phrase_ref = dict[int, set[int]]()
    for phrase, end_positions in duplicated_phrases.items():
        add_duplicated_phrases(duplicated_ranges,
                               duplicated_reversed_ranges,
                               duplicated_words,
                               touched_positions, line, phrase,
                               end_positions,
                               prev_phrase_ref,
                               concatenated_phrases,
                               concatenated_prev_phrase_ref)

    # while len(concatenated_phrases) < len(duplicated_phrases):
    #     old_duplicated_phrases = duplicated_phrases
    #     duplicated_phrases = concatenated_phrases
    #     prev_phrase_ref = concatenated_prev_phrase_ref
    #     concatenated_phrases = dict[str, set[int]]()
    #     concatenated_prev_phrase_ref = dict[int, set[int]]()
    #     for phrase, end_positions in duplicated_phrases.items():
    #         add_duplicated_phrases(duplicated_ranges, duplicated_reversed_ranges,
    #                                duplicated_words, touched_positions,
    #                                line, phrase, end_positions, prev_phrase_ref,
    #                                concatenated_phrases, concatenated_prev_phrase_ref)

    return


def add_duplicated_phrases(duplicated_ranges: dict[int, int], duplicated_reversed_ranges: dict[int, int],
                           duplicated_words: dict[str, set[int]], touched_positions: set[int],
                           line: list[str], phrase: str, end_positions: set[int],
                           prev_phrase_ref: dict[int, set[int]],
                           concatenated_phrases: dict[str, set[int]],
                           concatenated_prev_phrase_ref: dict[int, set[int]]):
    if len(end_positions) > 1:
        for end_position in end_positions:
            start_position = end_position - (len(phrase) - 1)
            add_duplicated_phrase(duplicated_ranges, duplicated_reversed_ranges, duplicated_words,
                                  touched_positions, line, phrase, end_position, prev_phrase_ref)

            prev_phrase_end = start_position - 1
            if prev_phrase_end >= 0:
                prev_phrase_start = duplicated_reversed_ranges.get(prev_phrase_end)
                if not prev_phrase_start is None:
                    prev_phrase = get_word(line, prev_phrase_start, prev_phrase_end)
                    new_phrase = prev_phrase + phrase
                    concatenated_phrases.setdefault(new_phrase, set()).add(start_position)
                    concatenated_prev_phrase_ref.setdefault(end_position, set()).add(prev_phrase_end)


def add_duplicated_phrase(duplicated_ranges: dict[int, int], duplicated_reversed_ranges: dict[int, int],
                          duplicated_words: dict[str, set[int]], touched_positions: set[int], line: list[str],
                          phrase: str, end_position: int,
                          prev_phrase_ref: dict[int, set[int]]) -> int:
    start_position = end_position - (len(phrase) - 1)
    exists_phrase_start_position = duplicated_reversed_ranges.get(end_position)

    is_added = False
    if exists_phrase_start_position is None or exists_phrase_start_position > start_position:
        if not exists_phrase_start_position is None:
            # delete old phrase
            delete_from_ranges(duplicated_ranges, duplicated_reversed_ranges, duplicated_words, touched_positions, line,
                               exists_phrase_start_position, end_position)
        pass
        is_added, real_phrase = add_to_ranges(duplicated_ranges, duplicated_reversed_ranges, duplicated_words,
                                              touched_positions, line, phrase,
                                              start_position,
                                              end_position)
        pass
    else:
        pass

    # remove previous obsolete phrases
    prev_phrase_ends = prev_phrase_ref.get(end_position)
    if not prev_phrase_ends is None:
        for prev_phrase_end in prev_phrase_ends:
            prev_phrase_start = duplicated_reversed_ranges.get(prev_phrase_end)
            if not prev_phrase_start is None:
                # new phrase must full include old phrase for deleting the last one
                is_inside_new_phrase = start_position <= prev_phrase_start and prev_phrase_end < end_position
                if is_inside_new_phrase:
                    delete_from_ranges(duplicated_ranges, duplicated_reversed_ranges, duplicated_words,
                                       touched_positions,
                                       line, prev_phrase_start, prev_phrase_end)
        del prev_phrase_ref[end_position]
    return start_position


def add_to_ranges(duplicated_ranges: dict[int, int], duplicated_reversed_ranges: dict[int, int],
                  duplicated_words: dict[str, set[int]], touched_positions: set[int], line: list[str], phrase: str,
                  position_start: int,
                  position_end: int) -> tuple[bool, str]:
    old_end_position = duplicated_ranges.get(position_start)
    if old_end_position is None or old_end_position < position_end:
        if not old_end_position is None:
            delete_from_ranges(duplicated_ranges, duplicated_reversed_ranges, duplicated_words, touched_positions, line,
                               position_start, old_end_position)
        duplicated_reversed_ranges[position_end] = position_start
        duplicated_ranges[position_start] = position_end
        duplicated_words.setdefault(phrase, set()).add(position_start)
        touched_positions.update(range(position_start, position_end + 1))
        return True, get_word(line, position_start, position_end)
    else:
        longer_phrase = get_word(line, position_start, old_end_position)
        return False, longer_phrase


def add_token(token: str, line: list[str]):
    line.append(token)


def add_check_duplicate_tokens(check_duplicate_tokens: dict[str, list[int]], line: list[str], token: str):
    check_duplicate_tokens.setdefault(token, []).append(len(line) - 1)


def get_word(line: list[str], start: int, end: int) -> str:
    return "".join(line[start:end + 1])


def delete_from_ranges(duplicated_ranges: dict[int, int],
                       duplicated_reversed_ranges: dict[int, int],
                       duplicated_words: dict[str, set[int]],
                       touched_positions: set[int],
                       line: list[str],
                       position_start: int, position_end: int):
    pass
    phrase = get_word(line, position_start, position_end)
    delete_word(duplicated_ranges, duplicated_reversed_ranges, duplicated_words, touched_positions, line, phrase,
                position_start)
    clear_ranges(duplicated_ranges, duplicated_reversed_ranges, touched_positions, position_start, position_end)
    pass


def clear_ranges(duplicated_ranges: dict[int, int], duplicated_reversed_ranges: dict[int, int],
                 touched_positions: set[int], position_start: int, position_end: int):
    del duplicated_ranges[position_start]
    del duplicated_reversed_ranges[position_end]
    touched_positions.difference_update(range(position_start, position_end + 1))


def delete_word(duplicated_ranges: dict[int, int],
                duplicated_reversed_ranges: dict[int, int],
                duplicated_words: dict[str, set[int]],
                touched_positions: set[int],
                line: list[str],
                phrase: str, position_start: int):
    phrase_positions = duplicated_words.get(phrase)
    if phrase_positions:
        phrase_positions.remove(position_start)
        if len(phrase_positions) == 1:
            position_start = next(iter(phrase_positions))
            position_end = position_start + len(phrase) - 1
            delete_from_ranges(duplicated_ranges, duplicated_reversed_ranges, duplicated_words,
                               touched_positions, line, position_start, position_end)
        elif len(phrase_positions) == 0:
            del duplicated_words[phrase]


def del_phrase(expected_prev_step_phrase_positions: set[int], line_phrases: dict[str, set[int]],
               line_phrases_back: dict[int, set[str]], prev_step_phrase: str):
    del line_phrases[prev_step_phrase]
    del_from_line_phrase_back(prev_step_phrase, expected_prev_step_phrase_positions, line_phrases_back)


def del_from_line_phrase_back(phrase: str, positions: set[int], line_phrases_back: dict[int, set[str]]):
    for on_del_position in positions:
        phrases_on_position = line_phrases_back.get(on_del_position)
        if phrases_on_position:
            phrases_on_position.remove(phrase)
            if len(phrases_on_position) == 0:
                del line_phrases_back[on_del_position]


def updat_back_ref(back_ref: dict[int, set[int]], phrase: str, position: int):
    i = position + len(phrase) - 1
    while i >= position:
        start_positions = back_ref.get(i, set[int]())
        start_positions.add(position)
        back_ref[i] = start_positions
        i -= 1


def get_prev_phrases(line: list[str], prev_phrase_starts: set[int] | list[Any], prev_token_position: int) -> dict[
    str, int]:
    return {"".join(line[prev_phrase_start:prev_token_position + 1]): prev_phrase_start for
            i, prev_phrase_start in enumerate(prev_phrase_starts)}


class Phrase:
    def __init__(self, strat_duplicates_detect_from: int = 500,
                 in_line_duplicates_rate: float = 0.8, in_line_end_rate: float = .99):
        self.tokens: list[str] = []
        self.lines: list[str] = []
        self.current_line: list[str] = []
        self.check_duplicate_tokens: dict[str, list[int]] = {}
        self.lines_unique: dict[str, list[int]] = {}
        self.lines_duplicated_times: dict[int, set[str]] = {}
        self.duplicated_reversed_ranges = dict[int, int]()
        self.duplicated_ranges = dict[int, int]()
        self.duplicated_positions = set[int]()
        self.duplicated_words = dict[str, set[int]]()
        self.in_line_duplicates_detect_start_amount = strat_duplicates_detect_from
        self.in_line_duplicates_rate = in_line_duplicates_rate
        self.in_line_end_rate = in_line_end_rate

    @property
    def full(self):
        join = "".join(self.tokens)
        return join

    def add_token(self, token: str) -> str | None:
        self.tokens.append(token)

        prev_token = self.tokens[-1]
        if prev_token == token:
            i = 1
            for prev_token in reversed(self.tokens[:-1]):
                if prev_token != token:
                    break
                i += 1
                if i >= DUPLICATED_TOKENS_LIMIT:
                    raise LoopError(message="Duplicate token looks like the model is in a loop", payload=token)

        if token != '\n':
            add_token(token, self.current_line)
            if len(self.current_line) > self.in_line_duplicates_detect_start_amount:
                duplicates_check_tail = self.current_line[self.in_line_duplicates_detect_start_amount:]
                add_check_duplicate_tokens(self.check_duplicate_tokens, duplicates_check_tail, token)

                register_to_check_duplicates(token, duplicates_check_tail,
                                             self.check_duplicate_tokens,
                                             self.duplicated_reversed_ranges,
                                             self.duplicated_ranges,
                                             self.duplicated_words,
                                             self.duplicated_positions)

                total_tokens = len(duplicates_check_tail)
                duplicated_tokens = len(self.duplicated_positions)
                merged_ranges, calculated_merged_part_size, last_positions = merge_intersect_ranges(
                    self.duplicated_ranges)

                last_part_position = last_positions[-1] if last_positions else None

                end_rate = (last_part_position or 0) / total_tokens

                duplicated_tokens_in_last_part = calculated_merged_part_size[
                    last_part_position] if last_part_position else 0
                prev_last_part_position = last_positions[-2] if len(last_positions) > 1 else None
                duplicated_tokens_in_prev_last_part = calculated_merged_part_size[
                    prev_last_part_position] if prev_last_part_position else 0

                # visual = visualize_duplicated_parts(duplicates_check_tail, merged_ranges)

                max_part_rate = duplicated_tokens / total_tokens
                last_part_rate = (duplicated_tokens_in_last_part + duplicated_tokens_in_prev_last_part) / total_tokens
                if end_rate >= self.in_line_end_rate and last_part_rate > self.in_line_duplicates_rate:
                    duplicated_payload = "".join(duplicates_check_tail)
                    raise LoopError(message="Duplicate fragments looks like the model is in a loop",
                                    payload=duplicated_payload)

            return None
        else:
            current_line = self.current_line
            current_line_str = "".join(current_line)
            self.lines.append(current_line_str)
            lines_amount = len(self.lines)
            current_line_positions = self.lines_unique.get(current_line_str, [])
            duplicated_amount = len(current_line_positions)
            duplicated_time_lines: set[str] | None = self.lines_duplicated_times.get(
                duplicated_amount) if duplicated_amount > 0 else None
            if duplicated_time_lines is not None:
                duplicated_time_lines.remove(current_line_str)
                if len(duplicated_time_lines) > 0:
                    self.lines_duplicated_times[duplicated_amount] = duplicated_time_lines
                else:
                    del self.lines_duplicated_times[duplicated_amount]

            current_line_positions.append(lines_amount)

            duplicated = len(current_line_positions)
            duplicated_time_lines: set[str] = self.lines_duplicated_times.get(duplicated) or set()
            duplicated_time_lines.add(current_line_str)
            self.lines_duplicated_times[duplicated] = duplicated_time_lines

            self.lines_unique[current_line_str] = current_line_positions

            positions = current_line_positions
            duplicated_phrase_revert = [current_line_str]
            if len(positions) >= DUPLICATED_LINES_THRESHOLD:
                prev_lines_position_unique = dict[int, int]()
                prev_line_step = 1
                stop = False
                touched_line_positions = set[int]()
                cycle_start = None
                cycle_end = None
                while not stop:
                    prev_lines_unique = set[str]()
                    prev_lines = list[str]()
                    prev_line_num = list[int]()

                    for line_position in reversed(positions):
                        touched_line_positions.add(line_position)
                        prev_line_position = line_position - prev_line_step
                        if prev_line_position <= 0:
                            stop = True
                            break

                        prev_line = self.lines[prev_line_position - 1]
                        prev_lines_position_unique[line_position] = prev_line_position

                        if prev_line_position in touched_line_positions:
                            stop = True
                            cycle_start = prev_line_position
                            cycle_end = line_position
                            break

                        touched_line_positions.add(prev_line_position)

                        prev_line_num.append(len(prev_lines))
                        if len(prev_lines) == 0 or not (prev_line in prev_lines_unique):
                            prev_lines_unique.add(prev_line)
                            prev_lines.append(prev_line)

                    if len(prev_lines) == 1:
                        # prev fully duplicated so
                        duplicated_phrase_revert.append(prev_lines[0])
                        prev_line_step += 2
                    else:
                        stop = True

                if cycle_start and cycle_end:
                    cycled_phrase = "\n".join([self.lines[fi - 1] for fi in range(cycle_start, cycle_end + 1)])
                    raise LoopError(message="Cycled phrase looks like the model is in a loop", payload=cycled_phrase)
                else:
                    duplicated_phrase = "\n".join(reversed(duplicated_phrase_revert))
                    log.debug(f"duplicated phrase '{duplicated_phrase}', times {len(positions)}")

        duplicated_lines_amount = lines_amount - len(self.lines_unique)  # len(duplicated_lines)
        duplicated_rate = duplicated_lines_amount / lines_amount
        if duplicated_rate >= DUPLICATED_LINES_RATE_LIMIT and duplicated_lines_amount >= DUPLICATED_LINES_LIMIT:
            payload = "".join(list(self.lines_unique.keys()))
            raise LoopError(message="Duplicate lines looks like the model is in a loop", payload=payload)

        self.clean_current_line()
        return current_line_str

    def append_token_positions(self, token: str) -> list[int]:
        current_line_tokens = self.check_duplicate_tokens
        token_positions = current_line_tokens.get(token, [])
        token_positions.append(len(self.current_line) - 1)
        current_line_tokens[token] = token_positions
        return token_positions

    def clean_current_line(self):
        self.current_line.clear()
        self.check_duplicate_tokens.clear()
