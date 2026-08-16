import logging

from agent.inference.loop_error import LoopError, ERROR_APPEARS_TO_BE_A_LOOP

DEFAULT_DUPLICATED_TOKENS_LIMIT = 100

IGNORE_DUPLICATED_PARTS_SIZE = 10

INIT_STEP = 1

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
                                    duplicated_words, token_positions, islands, duplicates_islands_reversed, phrase,
                                    position_end)
        else:
            for position_end in position_ends:
                add_duplicated_pair(line, single_tokens, duplicate_ranges, duplicated_ranges_reversed,
                                    duplicated_words, token_positions, islands, duplicates_islands_reversed, phrase,
                                    position_end)
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
                        touched_positions: set[int],
                        islands: dict[int, int], duplicates_islands_reversed: dict[int, int], phrase: str,
                        phrase_end: int):
    phrase_start = phrase_end - (len(phrase) - 1)
    phrase_end_old = duplicate_ranges.get(phrase_start)
    if not phrase_end_old is None and phrase_end_old != phrase_end:
        delete_from_ranges(duplicate_ranges, duplicated_ranges_reversed, duplicated_words, touched_positions, line,
                           phrase_start, phrase_end_old)
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


def find_duplicated_with_longest_last(last_part_start: int | None, last_word: str, start_positions: list[int],
                                      duplicates_check_tail: list[str], duplicated_ranges_reversed: dict[int, int],
                                      duplicated_words: dict[str, set[int]]) -> tuple[str | None, list[int]]:
    merged_word: str | None = None
    end_positions = list[int]()
    for start_position in start_positions:
        prev_word_start = duplicated_ranges_reversed.get(start_position)
        if prev_word_start and prev_word_start >= last_part_start:
            prev_word = "".join(duplicates_check_tail[prev_word_start: start_position + 1])
            if duplicated_words.get(prev_word):
                end_position = start_position + len(last_word) - 1
                join = "".join(duplicates_check_tail[prev_word_start: end_position + 1])
                if merged_word is None:
                    merged_word = join
                    end_positions.append(end_position)
                elif join == merged_word:
                    end_positions.append(end_position)

    while not merged_word is None and len(start_positions) > 1:
        new_merged_word: str | None = None
        new_end_positions = list[int]()
        shift = (len(merged_word) - 1)
        for end_position in end_positions:
            start_position = end_position - shift
            prev_word_start = duplicated_ranges_reversed.get(start_position)
            if prev_word_start in end_positions:
                # loop
                break
            elif prev_word_start and prev_word_start >= last_part_start:
                prev_word = "".join(duplicates_check_tail[prev_word_start: start_position + 1])
                if duplicated_words.get(prev_word):
                    join = "".join(duplicates_check_tail[prev_word_start: end_position + 1])
                    if not new_merged_word:
                        new_merged_word = join
                        new_end_positions.append(end_position)
                    elif join == new_merged_word:
                        new_end_positions.append(end_position)
                    else:
                        pass

        if len(new_end_positions) > 1:
            merged_word = new_merged_word
            end_positions = new_end_positions
        elif len(new_end_positions) == 0:
            # loop
            break
        else:
            merged_word = None
            end_positions = []
            break

    return merged_word, end_positions


class Phrase:
    def __init__(self, strat_duplicates_detect_from: int = 500, last_part_duplicates_rate: float = 0.5,
                 last_subpart_duplicates_rate: float = 0.49, last_subpart_end_line_delta_rate: float = 0.0025,
                 duplicated_tokens_limit=DEFAULT_DUPLICATED_TOKENS_LIMIT, duplicated_lines_rate_limit=0.6,
                 duplicated_lines_limit=50,
                 duplicated_lines_threshold=5):
        self.tokens: list[str] = []
        self.lines: list[str] = []
        self.lines_unique: dict[str, list[int]] = {}
        self.lines_duplicated_times: dict[int, set[str]] = {}

        self.current_line: list[str] = []
        self.current_line_duplicated_count: int = 0
        self.current_line_has_no_pair_tokens: dict[str, set[int]] = {}
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

        self.duplicated_tokens_limit = duplicated_tokens_limit
        self.duplicated_lines_rate_limit = duplicated_lines_rate_limit
        self.duplicated_lines_limit = duplicated_lines_limit
        self.duplicated_lines_threshold = duplicated_lines_threshold

    @property
    def full(self):
        join = "".join(self.tokens)
        return join

    def add_token(self, token: str) -> list[str]:
        added_lines = list[str]()
        if token == "":
            log.error(f"empty token")
            token = " "
        for letter in token:
            self.tokens.append(letter)

            prev_token = self.tokens[-1]
            if prev_token == letter:
                i = 1
                for prev_token in reversed(self.tokens[:-1]):
                    if prev_token != letter:
                        break
                    i += 1
                    if i >= self.duplicated_tokens_limit:
                        raise LoopError(payload=letter, message=f"Duplicated tokens (amount={i})")

            if letter != '\n':
                add_token(letter, self.current_line)
                if len(self.current_line) > self.in_line_duplicates_detect_start_amount:
                    duplicates_check_tail = self.current_line[self.in_line_duplicates_detect_start_amount:]
                    token_positions = self.current_line_has_no_pair_tokens
                    add_check_duplicate_tokens(token_positions, letter, len(duplicates_check_tail) - 1)

                    process_duplicate_pairs(letter, duplicates_check_tail,
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
                        log.debug(
                            f"duplicates detector: last_part_rate={last_part_rate}, last_island_rate={self.last_island_rate}")
                        self.last_island_rate = last_part_rate
                        last_sub_islands = layout_last_island(duplicates_check_tail, last_part_start, last_part_end)
                        # sub_island_sizes = get_island_sizes(last_sub_islands)

                        last_subpart_start, last_subpart_end = get_last_part_border(duplicates_check_tail,
                                                                                    last_sub_islands)
                        subpart_size = (last_subpart_end + 1) - last_subpart_start if (
                                not last_subpart_start is None and
                                not last_subpart_end is None) else 0

                        delta = total_tokens_amount - last_subpart_end if subpart_size else total_tokens_amount
                        delta_rate = delta / total_tokens_amount
                        total_tokens_amount = len(duplicates_check_tail)
                        last_part_rate2 = subpart_size / total_tokens_amount
                        log.debug(
                            f"duplicates detector: delta_rate={delta_rate}, last_part_rate2={last_part_rate2}")
                        if delta_rate <= self.last_subpart_end_line_delta_rate and last_part_rate2 >= self.last_subpart_duplicates_rate:
                            last_duplicated_range_start = self.duplicate_ranges_reversed[last_part_end]
                            last_word = "".join(duplicates_check_tail[last_duplicated_range_start: last_part_end + 1])
                            last_word_positions = self.duplicated_words[last_word]

                            start_positions = list(last_word_positions)
                            start_positions.sort(reverse=True)

                            longest_last_duplicated_word, end_positions = find_duplicated_with_longest_last(
                                last_part_start, last_word,
                                start_positions,
                                duplicates_check_tail,
                                self.duplicate_ranges_reversed,
                                self.duplicated_words)

                            if not longest_last_duplicated_word is None and len(end_positions) > 1:
                                duplicated_payload = "\n".join([longest_last_duplicated_word] * len(end_positions))
                                raise LoopError(payload=duplicated_payload, message=ERROR_APPEARS_TO_BE_A_LOOP)
                            else:
                                pass
            else:
                current_line = self.current_line
                current_line_str = "".join(current_line)
                lines = self.lines
                if lines and lines[-1] == current_line_str:
                    self.current_line_duplicated_count += 1
                else:
                    self.current_line_duplicated_count = 0
                lines.append(current_line_str)
                lines_amount = len(lines)
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

                start_positions = current_line_positions
                duplicated_phrase_revert = [current_line_str]
                len_start_positions = len(start_positions)
                if len_start_positions > 2 and len_start_positions >= self.duplicated_lines_threshold:
                    cycle_start = None
                    cycle_end = None
                    i = len_start_positions - 1
                    while i >= 0:
                        line_position = start_positions[i]
                        prev_line_position = start_positions[i - 1]
                        snapshot = lines[prev_line_position:line_position]
                        prev_prev_line_position = start_positions[i - 2]
                        snapshot2 = lines[prev_prev_line_position:prev_line_position]

                        if snapshot and snapshot2 and snapshot == snapshot2:
                            cycle_start = line_position
                            cycle_end = prev_line_position
                            break
                        i -= 1

                    if cycle_start and cycle_end:
                        cycled_phrase = "\n".join([lines[fi - 1] for fi in range(cycle_start, cycle_end + 1)])
                        payload = "\n".join(lines)
                        log.error(
                            f"cycled phrase detected:\npayload={payload}\ncycled_phras={cycled_phrase}\ncycle_start={cycle_start}, cycle_end={cycle_end}")
                        raise LoopError(payload=cycled_phrase, message="Cycled phrase detected")
                    else:
                        duplicated_phrase = "\n".join(reversed(duplicated_phrase_revert))
                        if len(duplicated_phrase.strip()) > 0:
                            log.debug(f"duplicated phrase '{duplicated_phrase}', times {len_start_positions}")

                # duplicated_lines_amount = self.current_line_duplicated_count + 1  # len(duplicated_lines)
                # duplicated_rate = duplicated_lines_amount / lines_amount
                # if duplicated_rate >= self.duplicated_lines_rate_limit and duplicated_lines_amount >= self.duplicated_lines_limit:
                #     payload = "\n".join(lines[-duplicated_lines_amount:])
                #     log.error(f"duplicated lines detected:\n{payload}")
                #     raise LoopError(payload=payload,
                #                     message=f"Duplicated lines detected (amount={duplicated_lines_amount})")

                self.clean_current_line()
                added_lines.append(current_line_str)
        return added_lines

    def clean_current_line(self):
        self.current_line.clear()
        self.current_line_has_no_pair_tokens.clear()
        self.duplicate_ranges_reversed.clear()
        self.duplicate_ranges.clear()
        self.duplicated_words.clear()
        self.duplicates_islands.clear()
        self.duplicates_islands_reversed.clear()
        self.last_island_rate = 0.0
        self.start_time = None
