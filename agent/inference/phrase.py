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

DUPLICATES_IN_LINE_MIN_LINE_LEN = 500
DUPLICATES_IN_LINE_RATE = 0.4

log = logging.getLogger(__name__)


def get_ranges_with_duplicates_started_by_token(token: str, line_tokens: dict[str, list[int]],
                                                line: list[str]) -> dict[int, int]:
    token_positions = line_tokens[token]

    if len(token_positions) <= 1:
        return {}
    len_line = len(line)
    token_repeats = dict[int, str]()
    repeated_tokens_start = dict[int, int]()
    check_token_positions = token_positions[:]
    check_token_positions_next_round = []
    duplicate_ranges = dict[int, int]()
    step = INIT_STEP
    pivot = 0
    stop = len(check_token_positions) <= 1
    while not stop:
        if pivot >= len(check_token_positions):
            if check_token_positions_next_round:
                check_token_positions = check_token_positions_next_round
                check_token_positions_next_round = []
                step = INIT_STEP
                pivot = 0
                continue
            else:
                break

        token_position = check_token_positions[pivot]
        next_token_position = token_position + step
        if next_token_position >= len_line:
            pivot += 1
            continue
        next_token = line[next_token_position]
        next_token_positions = line_tokens[next_token]
        if len(next_token_positions) <= 1:
            pivot += 1
            continue
        elif next_token == token:
            cycled = True
            pivot += 1
            if step == INIT_STEP:
                repeated_token_start = repeated_tokens_start.get(token_position - 1)
                already_repeated = False
                if repeated_token_start:
                    repeated_token = token_repeats.get(repeated_token_start)
                    already_repeated = repeated_token == token
                    if already_repeated:
                        old_step = token_position - repeated_token_start
                        duplicate_ranges[repeated_token_start] = old_step + step + 1
                        repeated_tokens_start[token_position] = repeated_token_start

                if not already_repeated:
                    duplicate_ranges[token_position] = step + 1

                    token_repeats[token_position] = token
                    repeated_tokens_start[token_position] = token_position
                    repeated_tokens_start[token_position + 1] = token_position
            else:
                duplicates_count = duplicate_ranges.get(next_token_position)
                step = duplicates_count if duplicates_count else INIT_STEP
            continue

        move_step = False
        # loop tail
        i = pivot + 1
        check_token_positions_next_step = check_token_positions[:pivot + 1]
        len_check_token_positions = len(check_token_positions)
        while i < len_check_token_positions:
            token_position_next = check_token_positions[i]
            next_token_position_next = token_position_next + step
            if next_token_position_next not in next_token_positions:
                check_token_positions_next_round.append(token_position_next)
            else:
                check_token_positions_next_step.append(token_position_next)
                old_range_next = duplicate_ranges.get(token_position_next, 0)
                duplicate_ranges[token_position_next] = max(step + 1, old_range_next)
                move_step = True
            i += 1
        if move_step:
            old_range = duplicate_ranges.get(token_position, 0)
            duplicate_ranges[token_position] = max(step + 1, old_range)

            check_token_positions = check_token_positions_next_step
            step = duplicate_ranges[token_position]
        elif check_token_positions_next_round:
            check_token_positions = check_token_positions_next_round
            check_token_positions_next_round = []
            step = INIT_STEP
            pivot = 0
        else:
            stop = True
    return merge_intersect_ranges(line, duplicate_ranges)


def merge_ranges(duplicate_ranges: dict[int, int], filter_leq: int | None = None) -> dict[int, int]:
    bitset = 0
    last = 0
    for start, amount in duplicate_ranges.items():
        last = max(last, start + amount)
        for i in range(amount):
            start_i = start + i
            bitset |= 1 << (start_i)

    result_ranges = dict[int, int]()
    start = None
    for i in range(last + 1):
        is_in_range = bool(bitset & (1 << i))
        if is_in_range:
            if start is None:
                start = i
        elif start:
            finish = i
            amount = (finish - start)
            if not filter_leq or amount > filter_leq:
                result_ranges[start] = amount
            start = None
    return result_ranges


def merge_intersect_ranges(line: list[str], duplicate_ranges: dict[int, int]) -> dict[
    int, int]:
    bitset = 0
    last = 0
    for start, amount in duplicate_ranges.items():

        last = max(last, start + amount)
        for i in range(amount):
            start_i = start + i
            bitset |= 1 << (start_i)

    result_ranges = dict[int, int]()
    start: int | None = None
    expected_finish: int | None = None
    for i in range(last + 1):
        is_in_range = bool(bitset & (1 << i))
        if is_in_range:
            if start is None:
                start = i
                amount = duplicate_ranges.get(i)
                expected_finish = start + amount
            else:
                if i == expected_finish:
                    amount = (i - start)
                    result_ranges[start] = amount

                    start = i
                    amount = duplicate_ranges.get(i)
                    expected_finish = start + amount
                else:
                    intersect_amount = duplicate_ranges.get(i)
                    if intersect_amount:
                        intersect_expected_finish = i + intersect_amount
                        new_expected_finish = max(expected_finish, intersect_expected_finish)
                        expected_finish = new_expected_finish
        elif start:
            amount = (i - start)
            result_ranges[start] = amount
            start = None
            expected_finish = None
    return result_ranges


def filter_ranges(duplicate_ranges: dict[int, int], filter_leq: int | None = None) -> dict[int, int]:
    if filter_leq:
        for start in list(duplicate_ranges.keys()):
            amount = duplicate_ranges[start]
            if amount <= filter_leq:
                del duplicate_ranges[start]

    return duplicate_ranges


def visualize_reversed_ranges(line: list[str], reversed_ranges: dict[int, int]) -> str:
    duplicated_ranges = reverse(reversed_ranges)
    return visualize_duplicated_parts(line, duplicated_ranges)


def reverse(reversed_ranges: dict[int, int]) -> dict[int, int]:
    duplicated_ranges = dict[int, int]()
    for end, start in reversed_ranges.items():
        exists_end = duplicated_ranges.get(start)
        if exists_end is None:
            duplicated_ranges[start] = end
        else:
            duplicated_ranges[start] = max(end, exists_end)
    return duplicated_ranges


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


def add_token_to_line(token: str, line: list[str], line_tokens: dict[str, list[int]],
                      duplicated_reversed_ranges: dict[int, int],
                      duplicated_ranges: dict[int, int],
                      duplicated_words: dict[str, set[int]],
                      duplicated_positions: set[int]) -> list[str]:
    line.append(token)
    token_positions = line_tokens.get(token, [])
    token_positions.append(len(line) - 1)
    line_tokens[token] = token_positions

    if len(token_positions) <= 1:
        return line

    exists_phrases = dict[str, int]()
    new_phrases = dict[str, int | None]()
    new_phrases_prev_step_position = dict[int, int]()
    new_phrases_size = set[int]()
    duplicated_phrases = dict[str, set[int]]()
    touched_positions = set[int]()
    for i, token_position in enumerate(token_positions):
        exists_phrase_start = duplicated_reversed_ranges.get(token_position)
        if not exists_phrase_start is None:
            phrase_exists = get_word(line, exists_phrase_start, token_position)
            exists_phrases[phrase_exists] = token_position
            # touched_positions.update(range(exists_phrase_start, token_position + 1))
            continue

        prev_token_position = token_position - 1
        next_token_position = token_position + 1
        current_token_phrase = token

        if token_position <= 0:
            continue
        if prev_token_position < 0:
            continue
        if prev_token_position in touched_positions:
            # intersection
            continue

        next_step_phrase_start = duplicated_reversed_ranges.get(next_token_position)
        if not next_step_phrase_start is None:
            next_prase_len = next_token_position - next_step_phrase_start + 1
            if next_prase_len > 1:
                next_phrase = get_word(line, next_step_phrase_start, next_prase_len)
                # already inside another word, skip
                continue

        next_step_phrase_end = duplicated_ranges.get(prev_token_position)
        if not next_step_phrase_end is None:
            # next_phrases = [get_word(line, prev_token_position, next_step_phrase_end) for next_step_phrase_end in
            #                 next_step_phrase_ends]
            next_phrase = "".join(line[prev_token_position:next_step_phrase_end + 1])
            # already inside another word, skip
            continue

        next_step_phrase_end = duplicated_ranges.get(token_position)
        if not next_step_phrase_end is None:
            # next_phrases = [get_word(line, token_position, next_step_phrase_end) for next_step_phrase_end in
            #                 next_step_phrase_ends]
            next_phrase = "".join(line[token_position:next_step_phrase_end + 1])
            # already inside another word, skip
            continue

        if i == len(token_positions) - 1:
            # find in exists actually only for the last token
            exists_phrase_position: int | None = None
            max_exist_phrase_size: int | None = None
            longest_exists_phrase: str | None = None
            for exist_phrase in exists_phrases:
                exist_phrase_size = len(exist_phrase)
                possible_phrase = get_word(line, (token_position - (exist_phrase_size - 1)), token_position)
                p = exists_phrases.get(possible_phrase)
                if not p is None:
                    if max_exist_phrase_size is None or exist_phrase_size > max_exist_phrase_size:
                        max_exist_phrase_size = exist_phrase_size
                        exists_phrase_position = p
                        longest_exists_phrase = exist_phrase

            if not longest_exists_phrase is None:
                duplicated_phrase_positions = duplicated_phrases.setdefault(longest_exists_phrase, set())
                duplicated_phrase_positions.add(exists_phrase_position)
                duplicated_phrase_positions.add(token_position)
                continue

        prev_phrase_start = duplicated_reversed_ranges.get(prev_token_position)
        phrase_expected_default = get_word(line, prev_token_position, token_position)
        expected_phrase_by_prev_step_position = {phrase_expected_default: None}
        if not prev_phrase_start is None:
            prev_tokens_phrase = get_word(line, prev_phrase_start, prev_token_position)
            expected_token_phrase = prev_tokens_phrase + current_token_phrase
            expected_phrase_by_prev_step_position[expected_token_phrase] = prev_token_position
            for phrase_size in new_phrases_size:
                if phrase_size > max(len(expected_token_phrase), len(prev_tokens_phrase)):
                    possible_start_position = token_position - (phrase_size - 1)
                    possible_prev_token_phrase = get_word(line, possible_start_position, prev_token_position)
                    expected_phrase_by_prev_step_position[
                        possible_prev_token_phrase + current_token_phrase] = prev_token_position

        # if token_position in duplicated_positions:
        #     continue

        duplicated_phrase_per_position = dict[int, str]()
        for expected_phrase, prev_phrase_position in expected_phrase_by_prev_step_position.items():
            if not prev_phrase_position is None:
                new_phrases_prev_step_position[token_position] = prev_phrase_position
            already_set_position = new_phrases.get(expected_phrase)
            if already_set_position is None:
                new_phrases[expected_phrase] = token_position
                new_phrases_size.add(len(expected_phrase))
            else:
                touched_start = already_set_position - (len(expected_phrase) - 1)
                touched_positions.update(range(touched_start, already_set_position + 1))
                start = token_position - (len(expected_phrase) - 1)
                if not start in touched_positions:
                    duplicated_phrase_positions = duplicated_phrases.setdefault(expected_phrase, set())

                    step_phrase_prev = duplicated_phrase_per_position.get(already_set_position)
                    if step_phrase_prev is None or len(step_phrase_prev) < len(expected_phrase):
                        duplicated_phrase_per_position[already_set_position] = expected_phrase
                        duplicated_phrase_positions.add(already_set_position)

                        if not step_phrase_prev is None:
                            prev_pos = duplicated_phrases[step_phrase_prev]
                            prev_pos.remove(already_set_position)
                            if len(prev_pos) == 0:
                                del duplicated_phrases[step_phrase_prev]
                                pass

                    step_phrase_prev2 = duplicated_phrase_per_position.get(token_position)
                    if step_phrase_prev2 is None or len(step_phrase_prev2) < len(expected_phrase):
                        duplicated_phrase_per_position[token_position] = expected_phrase
                        duplicated_phrase_positions.add(token_position)
                        if not step_phrase_prev2 is None:
                            prev2_pos = duplicated_phrases[step_phrase_prev2]
                            prev2_pos.remove(token_position)
                            if len(prev2_pos) == 0:
                                del duplicated_phrases[step_phrase_prev2]
                                pass
                else:
                    # intersection in the step
                    pass

    for phrase, end_positions in duplicated_phrases.items():
        if len(end_positions) == 1:
            # error
            pass
        else:
            for end_position in end_positions:
                start_position = end_position - (len(phrase) - 1)
                prev_phrase_end = new_phrases_prev_step_position.get(end_position)

                prev_phrase_start = duplicated_reversed_ranges.get(
                    prev_phrase_end) if not prev_phrase_end is None else None

                phrase_new = get_word(line, start_position, end_position)

                exists_start_phrase_position = duplicated_reversed_ranges.get(end_position)

                is_intersected = False
                if not prev_phrase_start is None and not prev_phrase_end is None:
                    phrase_old = get_word(line, prev_phrase_start, prev_phrase_end)
                    # new phrase must full include old phrase for deleting the last one
                    if start_position <= prev_phrase_start and end_position > prev_phrase_end:
                        delete_from_ranges(duplicated_ranges, duplicated_reversed_ranges, prev_phrase_start,
                                           prev_phrase_end)
                        word_position_starts = duplicated_words.get(phrase_old, set())
                        if word_position_starts:
                            word_position_starts.remove(prev_phrase_start)
                        if len(word_position_starts) <= 1:
                            for position_start in word_position_starts:
                                position_end = position_start + len(phrase_old) - 1
                                if position_end is None:
                                    # error
                                    pass
                                else:
                                    phrase_no_more_duplicated = get_word(line, position_start, position_end)
                                    if phrase_no_more_duplicated != phrase_old:
                                        # error
                                        pass
                                    else:
                                        delete_from_ranges(duplicated_ranges, duplicated_reversed_ranges,
                                                           position_start,
                                                           position_end)
                            del duplicated_words[phrase_old]
                    else:
                        is_intersected = True

                if exists_start_phrase_position is None or exists_start_phrase_position > start_position:
                    exists_end_position = duplicated_ranges.get(start_position)
                    update_positions = True
                    if not exists_end_position is None:
                        if exists_end_position < end_position:
                            delete_from_ranges(duplicated_ranges, duplicated_reversed_ranges, start_position, exists_end_position)
                            phrase_exists = get_word(line, start_position, exists_end_position)
                            word_starts = duplicated_words.setdefault(phrase_exists, set())
                            word_starts.remove(start_position)
                            if len(word_starts) == 0:
                                del duplicated_words[phrase_exists]
                        else:
                            update_positions = False
                            exists_start = duplicated_reversed_ranges[exists_end_position]
                            if exists_start is None or exists_start != start_position:
                                pass

                    if update_positions:
                        duplicated_reversed_ranges[end_position] = start_position
                        duplicated_ranges[start_position] = end_position
                        duplicated_words.setdefault(phrase_new, set()).add(start_position)
                elif start_position == exists_start_phrase_position:
                    pass
                else:
                    # alarm, trying to erase long phrase by short or equals
                    pass


                duplicated_positions.update(range(start_position, end_position + 1))

    return line


def get_word(line: list[str], start: int, end: int) -> str:
    return "".join(line[start:end + 1])


def delete_from_ranges(duplicated_ranges: dict[int, int], duplicated_reversed_ranges: dict[int, int],
                       position_start: int, position_end: int):
    del duplicated_reversed_ranges[position_end]
    # ends = duplicated_ranges.setdefault(position_start, set())
    # ends.remove(position_end)
    # if len(ends) == 0:
    del duplicated_ranges[position_start]


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
    def __init__(self):
        self.tokens: list[str] = []
        self.lines: list[str] = []
        self.current_line: list[str] = []
        self.current_line_tokens: dict[str, list[int]] = {}
        self.lines_unique: dict[str, list[int]] = {}
        self.lines_duplicated_times: dict[int, set[str]] = {}

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
            current_line_tokens = self.current_line_tokens
            current_line = self.current_line

            current_line = add_token_to_line(token, current_line, current_line_tokens)

            self.current_line = current_line
            self.current_line_tokens = current_line_tokens

            if len(current_line) >= DUPLICATES_IN_LINE_MIN_LINE_LEN:
                duplicates = get_ranges_with_duplicates_started_by_token(token, current_line_tokens, current_line)

                total_tokens = len(current_line)
                max_duplicated_part_amount = 0
                max_duplicated_part_start = 0

                for start, amount in duplicates.items():
                    if amount > max_duplicated_part_amount:
                        max_duplicated_part_amount = amount
                        max_duplicated_part_start = start

                max_part_rate = max_duplicated_part_amount / total_tokens
                if max_part_rate > DUPLICATES_IN_LINE_RATE:
                    end = max_duplicated_part_start + max_duplicated_part_amount
                    duplicated_payload = "".join(current_line[
                                                     max_duplicated_part_start:end])
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
        current_line_tokens = self.current_line_tokens
        token_positions = current_line_tokens.get(token, [])
        token_positions.append(len(self.current_line) - 1)
        current_line_tokens[token] = token_positions
        return token_positions

    def clean_current_line(self):
        self.current_line.clear()
        self.current_line_tokens.clear()
