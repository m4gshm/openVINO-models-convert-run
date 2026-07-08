import logging
from typing import Any

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


def visualize_duplicate_parts(line: list[str], duplicated_ranges: dict[int, int]) -> str:
    only_duplicates_line: list[str] = []
    i = 0
    while i < len(line):
        amount = duplicated_ranges.get(i)
        if not amount:
            only_duplicates_line.append("-")
            i += 1
        else:
            only_duplicates_line.append(line[i])
            end = i + amount
            i += 1
            while i < end:
                intersect_amount = duplicated_ranges.get(i)
                if intersect_amount:
                    intersect_end = i + intersect_amount
                    end = max(end, intersect_end)
                only_duplicates_line.append(line[i])
                i += 1

    result = "".join(only_duplicates_line)
    return result


def get_duplicated_parts(line: list[str], duplicated_ranges: dict[int, int]) -> dict[str, list[int]]:
    parts = dict[str, list[int]]()
    for start, amount in duplicated_ranges.items():
        part = "".join(line[start:start + amount])
        positions = parts.get(part, [])
        positions.append(start)
        parts[part] = positions
    return parts


def add_token_to_line(token1: str, line: list[str], line_tokens: dict[str, list[int]],
                      line_phrases: dict[str, set[int]], line_phrases_back: dict[int, set[str]]) -> list[str]:
    line.append(token1)
    token_positions = line_tokens.get(token1, [])
    token_positions.append(len(line) - 1)
    line_tokens[token1] = token_positions

    if len(token_positions) <= 1:
        return line

    phrases_positions = dict[str, set[int]]()
    prev_step_phrase_mapping = dict[str, str]()

    phrases = dict[int, set[str]]()
    for token_position in token_positions:
        position_phrases = line_phrases_back.get(token_position)
        if not position_phrases:
            position_phrases = {token1}
            line_phrases_back[token_position] = position_phrases

        phrases[token_position] = position_phrases

    step = 1
    while token_positions:
        i = len(token_positions) - 1

        phrases_next_round = dict[str, set[int]]()
        while i >= 0:
            token_position: int = token_positions[i]
            prev_token_position = token_position - step
            if prev_token_position < 0:
                i -= 1
                continue

            token = line[token_position]
            prev_token = line[prev_token_position]

            pref_token_positions = line_tokens.get(prev_token, [])
            if len(pref_token_positions) <= 1:
                i -= 1
                continue

            prev_step_phrases = phrases.get(token_position)
            for prev_step_phrase in prev_step_phrases:
                phrase = prev_token + prev_step_phrase

                prev_step_phrase_mapping[phrase] = prev_step_phrase

                phrase_positions = phrases_positions.get(phrase, set[int]())
                phrase_positions.add(prev_token_position)
                phrases_positions[phrase] = phrase_positions

                next_round_positions = phrases_next_round.get(phrase, set[int]())
                next_round_positions.add(prev_token_position)

                phrases_next_round[phrase] = next_round_positions
            i -= 1

        token_positions_next_round = set[int]()
        for phrase in list(phrases_next_round.keys()):
            phrase_positions = phrases_next_round[phrase]
            if len(phrase_positions) <= 1:
                del phrases_next_round[phrase]
            else:
                prev_step_phrase = prev_step_phrase_mapping.get(phrase)
                positions = line_phrases.get(phrase, set[int]())

                prev_step_phrase_positions = line_phrases.get(prev_step_phrase) if prev_step_phrase else None
                expected_prev_step_phrase_positions = set[int]()
                on_delete_prev_phrase_variants_on_position = dict[str, set[int]]()
                for position in phrase_positions:
                    prev_step_phrase_position = position + step
                    expected_prev_step_phrase_positions.add(prev_step_phrase_position)

                    token_positions_next_round.add(position)
                    positions.add(position)

                    prev_phrase_variants_on_position = line_phrases_back.get(position)
                    if prev_phrase_variants_on_position:
                        for prev_phrase_variant_on_position in list(prev_phrase_variants_on_position):
                            if len(prev_phrase_variant_on_position) < len(phrase):
                                prev_phrase_variant_all_positions = line_phrases.get(prev_phrase_variant_on_position)

                                if prev_phrase_variant_all_positions and prev_phrase_variant_all_positions == phrase_positions:
                                    on_delete_prev_phrase_variants_on_position[prev_phrase_variant_on_position] = set(prev_phrase_variant_all_positions)

                                    # prev_phrase_variant_all_positions.remove(position)
                                    # if len(prev_phrase_variant_all_positions) == 0:
                                    #     del line_phrases[prev_phrase_variant_on_position]

                                # prev_phrase_variants_on_position.remove(prev_phrase_variant_on_position)

                        # del line_phrases_back[position]

                    # if prev_step_phrase and prev_step_phrase_positions:
                    #     prev_step_phrase_position = position + step
                    #     if prev_step_phrase_position in prev_step_phrase_positions:
                    #         prev_step_phrase_positions.remove(prev_step_phrase_position)
                    #     if len(prev_step_phrase_positions) == 0:
                    #         del line_phrases[prev_step_phrase]
                    #     prev_step_phrases_on_position = line_phrases_back.get(prev_step_phrase_position, set[str]())
                    #     prev_step_phrases_on_position.remove(prev_step_phrase)
                    #     if len(prev_step_phrases_on_position) == 0:
                    #         del line_phrases_back[prev_step_phrase_position]


                    phrases_on_position = line_phrases_back.get(position, set[str]())
                    phrases_on_position.add(phrase)
                    line_phrases_back[position] = phrases_on_position

                for on_del_phrase, on_del_positions in on_delete_prev_phrase_variants_on_position.items():
                    del line_phrases[on_del_phrase]
                    del_from_line_phrase_back(line_phrases_back, on_del_phrase, on_del_positions)

                line_phrases[phrase] = positions


                if prev_step_phrase and expected_prev_step_phrase_positions == prev_step_phrase_positions:
                    del line_phrases[prev_step_phrase]
                    del_from_line_phrase_back(line_phrases_back, prev_step_phrase, expected_prev_step_phrase_positions)


        token_positions = list(token_positions_next_round)
        for phrase, positions in phrases_next_round.items():
            for position in positions:
                p = phrases.get(position, set[str]())
                p.add(phrase)
                phrases[position] = p

    return line


def del_from_line_phrase_back(line_phrases_back: dict[int, set[str]], on_del_phrase: str, on_del_positions: set[int]):
    for on_del_position in on_del_positions:
        phrases_on_position = line_phrases_back.get(on_del_position)
        if phrases_on_position:
            phrases_on_position.remove(on_del_phrase)
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
