import logging

from agent.inference.loop_error import LoopError

IGNORE_DUPLICATED_PARTS_SIZE = 10

INIT_STEP = 1

DUPLICATED_LINES_THRESHOLD = 5
DUPLICATED_TOKEN_THRESHOLD = 10

DUPLICATED_TOKENS_LIMIT = 100
DUPLICATED_LINES_RATE_LIMIT = 0.5
DUPLICATED_LINES_LIMIT = 50

log = logging.getLogger(__name__)


def visualize_reversed_ranges(line: list[str], reversed_ranges: dict[int, int]) -> str:
    result: list[str] = ["-"] * len(line)
    i = len(line) - 1
    while i >= 0:
        start = reversed_ranges.get(i)
        if start is None:
            i -= 1
        else:
            while i > start:
                result[i] = line[i]
                i -= 1
            result[start] = line[start]
    return "".join(result)


def visualize_tokens(line: list[str], tokens: dict[str, set[int]]) -> str:
    result: list[str] = ["-"] * len(line)
    for token, positions in tokens.items():
        for position in positions:
            result[position] = line[position]

    return "".join(result)


def visualize_islands_reversed(line: list[str], duplicates_islands_reversed: dict[int, int]) -> str:
    symbols = ["-", "*", "&", "+", "#", "~"]
    current_symbol = 0
    result: list[str] = [" "] * len(line)
    for end, start in duplicates_islands_reversed.items():
        i = start
        while i <= end:
            result[i] = symbols[current_symbol % len(symbols)]
            i += 1
        current_symbol += 1
    return "".join(result)


def visualize_ranges(line: list[str], duplicate_ranges: dict[int, int]) -> str:
    result: list[str] = ["-"] * len(line)
    for start, end in duplicate_ranges.items():
        i = start
        while i <= end:
            result[i] = line[i]
            i += 1
    return "".join(result)


def process_duplicate_pairs(token: str,
                            line: list[str],
                            single_tokens: dict[str, set[int]],
                            duplicated_ranges_reversed: dict[int, int],
                            duplicate_ranges: dict[int, int],
                            duplicated_words: dict[str, set[int]],
                            islands: dict[int, int], duplicates_islands_reversed: dict[int, int]):
    token_positions = single_tokens[token]

    duplicated_phrases_active = dict[str, set[int]]()
    for i, token_position in enumerate(token_positions):
        prev_token_position = token_position - 1

        if token_position <= 0:
            continue
        if prev_token_position < 0:
            continue

        phrase = line[prev_token_position] + token
        duplicated_phrases_active.setdefault(phrase, set[int]()).add(token_position)

    for phrase, position_ends in duplicated_phrases_active.items():
        if len(position_ends) == 1:
            exists_positions = duplicated_words.get(phrase)
            if exists_positions:
                position_end = next(iter(position_ends))
                add_duplicated_pair(line, single_tokens, duplicate_ranges, duplicated_ranges_reversed,
                                    duplicated_words, islands, duplicates_islands_reversed, phrase, position_end)
        else:
            for position_end in position_ends:
                add_duplicated_pair(line, single_tokens, duplicate_ranges, duplicated_ranges_reversed,
                                    duplicated_words, islands, duplicates_islands_reversed, phrase, position_end)
    return


def layout_last_island(line: list[str], start: int, end: int) -> dict[int, int]:
    token_positions = dict[str, set[int]]()
    duplicate_reversed_ranges = dict[int, int]()
    duplicate_ranges = dict[int, int]()
    duplicated_words = dict[str, set[int]]()
    duplicates_islands = dict[int, int]()
    duplicates_islands_reversed = dict[int, int]()
    for i in range(start, end + 1):
        token = line[i]
        add_check_duplicate_tokens(token_positions, token, i)
        process_duplicate_pairs(token, line, token_positions, duplicate_reversed_ranges, duplicate_ranges,
                                duplicated_words, duplicates_islands, duplicates_islands_reversed)
    return duplicates_islands_reversed


def get_last_part_border(line: list[str], line_islands_reversed: dict[int, int] | None) -> tuple[
    int | None, int | None]:
    amount = len(line)
    last_part_start: int | None = None
    last_part_end: int | None = None
    threshold = amount - amount * 0.1
    i = amount
    if line_islands_reversed:
        while i >= threshold:
            last_part_end = i
            last_part_start = line_islands_reversed.get(i)
            if last_part_start is not None:
                break
            i -= 1
    return last_part_start, last_part_end


def get_island_sizes(line_islands_reversed: dict[int, int] | None) -> list[int]:
    return [(end - start + 1) for end, start in
            line_islands_reversed.items()] if line_islands_reversed is not None else []


def add_duplicated_pair(line: list[str], single_tokens: dict[str, set[int]], duplicate_ranges: dict[int, int],
                        duplicated_ranges_reversed: dict[int, int], duplicated_words: dict[str, set[int]],
                        islands: dict[int, int], duplicates_islands_reversed: dict[int, int], phrase: str,
                        phrase_end: int):
    phrase_start = phrase_end - (len(phrase) - 1)
    duplicated_ranges_reversed[phrase_end] = phrase_start
    duplicate_ranges[phrase_start] = phrase_end
    duplicated_words.setdefault(phrase, set[int]()).add(phrase_start)

    island_start = duplicates_islands_reversed.get(phrase_end)
    if island_start is None:
        island_start = phrase_start
        island_end = phrase_end

        is_left_extended = False
        is_right_extended = False

        # find left intersected island
        left_island_start = find_left_island(duplicates_islands_reversed, islands, phrase_start)
        if not left_island_start is None:
            island_start = left_island_start
            is_left_extended = True
        else:
            for left_phrase_end in [phrase_start, phrase_start - 1]:
                if left_phrase_end >= 0:
                    left_phrase_start = duplicated_ranges_reversed.get(left_phrase_end)
                    if not left_phrase_start is None:
                        left_island_start = find_left_island(duplicates_islands_reversed, islands, left_phrase_end)
                        if not left_island_start is None:
                            island_start = left_island_start
                            is_left_extended = True
                            break

        right_island_end = find_right_island(duplicates_islands_reversed, islands, line, phrase_end)

        if not right_island_end is None:
            island_end = right_island_end
            is_right_extended = True
        else:
            for right_phrase_start in [phrase_end, phrase_end + 1]:
                if right_phrase_start < len(line):
                    right_phrase_end = duplicate_ranges.get(right_phrase_start)
                    if not right_phrase_end is None:
                        right_island_end = find_right_island(duplicates_islands_reversed, islands, line,
                                                             right_phrase_end)
                        if not right_island_end is None:
                            island_end = right_island_end
                            is_right_extended = True
                            break

        if is_left_extended or is_right_extended:
            duplicates_islands_reversed[island_end] = island_start
            islands[island_start] = island_end
            pass
        else:
            has_left_phrase = False
            has_right_phrase = False
            # create new or check if it inside big island
            for left_phrase_end in [phrase_start, phrase_start - 1]:
                if left_phrase_end >= 0 and left_phrase_end in duplicated_ranges_reversed:
                    has_left_phrase = True
                    break

            for right_phrase_start in [phrase_end, phrase_end + 1]:
                if right_phrase_start < len(line) and right_phrase_start in duplicate_ranges:
                    has_right_phrase = True
                    break

            if not (has_left_phrase and has_right_phrase):
                duplicates_islands_reversed[island_end] = island_start
                islands[island_start] = island_end
                pass
            else:
                # already in island
                pass
    else:
        # already in island
        pass

    token = line[phrase_end]
    token_positions = single_tokens.get(token)
    if token_positions:
        removed_position = phrase_end
        token_positions.discard(removed_position)
        if len(token_positions) == 0:
            single_tokens.pop(token, None)


def find_left_island(duplicates_islands_reversed: dict[int, int], islands: dict[int, int],
                     phrase_start: int) -> int | None:
    for left_island_end in [phrase_start, phrase_start - 1]:
        if left_island_end >= 0:
            left_island_start = duplicates_islands_reversed.get(left_island_end)
            if not left_island_start is None:
                duplicates_islands_reversed.pop(left_island_end)
                islands.pop(left_island_start)
                return left_island_start
    return None


def find_right_island(duplicates_islands_reversed: dict[int, int], islands: dict[int, int], line: list[str],
                      phrase_end: int) -> int | None:
    for right_island_start in [phrase_end, phrase_end + 1]:
        if right_island_start < len(line):
            right_island_end = islands.get(right_island_start)
            if not right_island_end is None:
                duplicates_islands_reversed.pop(right_island_end)
                islands.pop(right_island_start)
                return right_island_end
    return None


def add_token(token: str, line: list[str]):
    line.append(token)


def add_check_duplicate_tokens(token_positions: dict[str, set[int]], token: str, position: int):
    token_positions.setdefault(token, set[int]()).add(position)


def get_word(line: list[str], start: int, end: int) -> str:
    return "".join(line[start:end + 1])


def delete_from_ranges(duplicate_ranges: dict[int, int],
                       duplicated_ranges_reversed: dict[int, int],
                       duplicated_words: dict[str, set[int]],
                       touched_positions: set[int],
                       line: list[str],
                       position_start: int, position_end: int):
    pass
    phrase = get_word(line, position_start, position_end)
    delete_word(duplicate_ranges, duplicated_ranges_reversed, duplicated_words, touched_positions, line, phrase,
                position_start)
    clear_ranges(duplicate_ranges, duplicated_ranges_reversed, touched_positions, position_start, position_end)
    pass


def clear_ranges(duplicate_ranges: dict[int, int], duplicated_ranges_reversed: dict[int, int],
                 touched_positions: set[int], position_start: int, position_end: int):
    del duplicate_ranges[position_start]
    del duplicated_ranges_reversed[position_end]
    touched_positions.difference_update(range(position_start, position_end + 1))


def delete_word(duplicate_ranges: dict[int, int],
                duplicated_ranges_reversed: dict[int, int],
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
            delete_from_ranges(duplicate_ranges, duplicated_ranges_reversed, duplicated_words,
                               touched_positions, line, position_start, position_end)
        elif len(phrase_positions) == 0:
            del duplicated_words[phrase]


class Phrase:
    def __init__(self, strat_duplicates_detect_from: int = 500, last_part_duplicates_rate: float = 0.5,
                 last_subpart_duplicates_rate: float = 0.49, last_subpart_end_line_delta_rate: float = 0.0025):
        self.tokens: list[str] = []
        self.lines: list[str] = []
        self.current_line: list[str] = []
        self.current_line_has_no_pair_tokens: dict[str, set[int]] = {}
        self.lines_unique: dict[str, list[int]] = {}
        self.lines_duplicated_times: dict[int, set[str]] = {}
        self.duplicate_ranges_reversed = dict[int, int]()
        self.duplicate_ranges = dict[int, int]()
        self.duplicated_words = dict[str, set[int]]()
        self.duplicates_islands = dict[int, int]()
        self.duplicates_islands_reversed = dict[int, int]()
        self.last_island_rate = 0.0
        self.in_line_duplicates_detect_start_amount = strat_duplicates_detect_from
        self.last_part_duplicates_rate = last_part_duplicates_rate
        self.last_subpart_duplicates_rate = last_subpart_duplicates_rate
        self.last_subpart_end_line_delta_rate = last_subpart_end_line_delta_rate

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
                token_positions = self.current_line_has_no_pair_tokens
                add_check_duplicate_tokens(token_positions, token, len(duplicates_check_tail) - 1)

                process_duplicate_pairs(token, duplicates_check_tail,
                                        token_positions,
                                        self.duplicate_ranges_reversed,
                                        self.duplicate_ranges,
                                        self.duplicated_words,
                                        self.duplicates_islands,
                                        self.duplicates_islands_reversed)

                last_part_start, last_part_end = get_last_part_border(duplicates_check_tail,
                                                                      self.duplicates_islands_reversed)
                last_part_size = (last_part_end + 1) - last_part_start if (not last_part_start is None and
                                                                           not last_part_end is None) else 0
                total_tokens_amount = len(duplicates_check_tail)
                last_part_rate = last_part_size / total_tokens_amount
                if last_part_rate > self.last_part_duplicates_rate and last_part_rate - self.last_island_rate > 0.01:
                    self.last_island_rate = last_part_rate
                    last_sub_islands = layout_last_island(duplicates_check_tail, last_part_start, last_part_end)
                    # sub_island_sizes = get_island_sizes(last_sub_islands)

                    last_subpart_start, last_subpart_end = get_last_part_border(duplicates_check_tail, last_sub_islands)
                    subpart_size = (last_subpart_end + 1) - last_subpart_start if (not last_subpart_start is None and
                                                                                   not last_subpart_end is None) else 0

                    delta = total_tokens_amount - last_subpart_end if subpart_size else total_tokens_amount
                    delta_rate = delta / total_tokens_amount
                    total_tokens_amount = len(duplicates_check_tail)
                    last_part_rate2 = subpart_size / total_tokens_amount
                    if delta_rate <= self.last_subpart_end_line_delta_rate and last_part_rate2 >= self.last_subpart_duplicates_rate:
                        duplicated_payload = "".join(duplicates_check_tail)
                        raise LoopError(message="Looks like generating infinity loop",
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

    def clean_current_line(self):
        self.current_line.clear()
        self.current_line_has_no_pair_tokens.clear()
