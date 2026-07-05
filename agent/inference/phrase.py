import logging

from agent.inference.loop_error import LoopError


INIT_STEP = 1

DUPLICATED_LINES_THRESHOLD = 5
DUPLICATED_TOKEN_THRESHOLD = 10

DUPLICATED_TOKENS_LIMIT = 100
DUPLICATED_LINES_RATE_LIMIT = 0.5
DUPLICATED_LINES_LIMIT = 50

DUPLICATES_IN_LINE_RATE = 0.2

log = logging.getLogger(__name__)


def get_ranges_with_duplicates_started_by_token(token: str, line_tokens: dict[str, list[int]],
                                                line: list[str]) -> dict[int, int] | None:
    token_positions = line_tokens[token]
    if len(token_positions) <= 1:
        return None
    len_line = len(line)
    cycled = False
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
            # check_token_positions.pop(pivot)
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
                duplicate_ranges[token_position] = step + 1
            else:
                # pivot must be moved right
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

    return merge_ranges(duplicate_ranges)


def merge_ranges(duplicate_ranges: dict[int, int]) -> dict[int, int]:
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
            result_ranges[start] = (finish - start)
            start = None
    return result_ranges


def add_token_to_current_line(token: str, current_line: list[str], current_line_tokens: dict[str, list[int]]) -> tuple[
    list[str], list[int]]:
    current_line += token
    token_positions = current_line_tokens.get(token, [])
    token_positions.append(len(current_line) - 1)
    current_line_tokens[token] = token_positions
    return current_line, token_positions


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
                    raise LoopError(message="duplicated token", payload=token)

        if token != '\n':
            current_line_tokens = self.current_line_tokens
            current_line = self.current_line

            current_line, token_positions = add_token_to_current_line(token, current_line, current_line_tokens)

            self.current_line = current_line
            self.current_line_tokens = current_line_tokens

            duplicates = get_ranges_with_duplicates_started_by_token(token, token_positions, current_line)
            
            total_tokens = len(current_line)
            max_duplicated_part_amount = 0
            max_duplicated_part_start = 0
            
            for start, amount in duplicates.items():
                if amount > max_duplicated_part_amount:
                    max_duplicated_part_amount = amount
                    max_duplicated_part_start = start

            max_part_rate = max_duplicated_part_amount / total_tokens
            if max_part_rate > DUPLICATES_IN_LINE_RATE:
                duplicated_payload = current_line[max_duplicated_part_start:max_duplicated_part_start+max_duplicated_part_amount]
                raise LoopError(message="duplicated line part", payload=duplicated_payload)
            
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
                    raise LoopError(message="cycled", payload=cycled_phrase)
                else:
                    duplicated_phrase = "\n".join(reversed(duplicated_phrase_revert))
                    log.debug(f"duplicated phrase '{duplicated_phrase}', times {len(positions)}")

        duplicated_lines_amount = lines_amount - len(self.lines_unique)  # len(duplicated_lines)
        duplicated_rate = duplicated_lines_amount / lines_amount
        if duplicated_rate >= DUPLICATED_LINES_RATE_LIMIT and duplicated_lines_amount >= DUPLICATED_LINES_LIMIT:
            payload = list(self.lines_unique.keys())
            raise LoopError(message="duplicated lines", payload=payload)

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
