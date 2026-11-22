"""Tests for the Phrase duplicate/loop detector and its visualization helpers."""
import cProfile
import json
import pstats
import unittest
from importlib.resources import files

from agent.inference.loop_error import LoopError
from agent.inference.phrase import DEFAULT_DUPLICATED_TOKENS_LIMIT, Phrase, \
    process_duplicate_pairs, visualize_ranges, add_token, \
    add_check_duplicate_tokens, visualize_reversed_ranges, visualize_tokens, visualize_islands_reversed, \
    get_last_part_border, layout_last_island, find_duplicated_with_longest_last

TEST_RESOURCES = "test_resources/phrase"


class PhraseTestCase(unittest.TestCase):
    """Unit tests of Phrase loop detection and duplicated-parts visualization."""

    def test_loop_tokens(self):
        """A run of identical tokens must raise LoopError."""
        repeated_string = "a" + ("b" * DEFAULT_DUPLICATED_TOKENS_LIMIT)

        phrase = Phrase()
        with self.assertRaises(LoopError):
            for token in repeated_string:
                phrase.add_token(token)

    def test_loop_lines(self):
        """Repeated duplicated lines must raise LoopError keeping the normal output."""
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "loop_messages.txt")
        loop_messages = loop_messages_file.read_text()

        phrase = Phrase()
        with self.assertRaises(LoopError):
            for token in loop_messages:
                phrase.add_token(token)

        self.assertEqual(('some normal output\n'
                          'first\n'
                          'second\n'
                          'first\n'
                          'second\n'
                          'first\n'
                          'second\n'
                          'first\n'
                          'second\n'
                          'first\n'), phrase.full)

    def test_loop_lines6(self):
        """Loop inside a tool call payload must raise LoopError with the duplicated fragment."""
        loop_tokens_file = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line6.json")
        loop_tokens_text = loop_tokens_file.read_text()
        loop_tokens_json = json.loads(loop_tokens_text)

        phrase = Phrase()
        with self.assertRaises(LoopError) as context:
            for token in loop_tokens_json:
                phrase.add_token(token)

        exception = context.exception
        error_message = exception.message
        self.assertEqual('Generated content appears to be a loop', error_message)
        # pylint: disable=line-too-long  # long literal expectations
        self.assertEqual('-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent', exception.payload)
        self.assertEqual(('<function=read_file>\n'
                          '<parameter=target_file>\n'
                          'C:/alex/github/m4gshm/distributed-transactions-practice/java/idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent-consumer-jdbc/../idempotent'),
                         phrase.full)

    def test_loop_lines_2(self):
        """Feed loop_messages_2 tokens: a three-line cycle raises on its third repeat."""
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "loop_messages_2.txt")
        loop_messages = loop_messages_file.read_text()

        phrase = Phrase()
        with self.assertRaises(LoopError) as context:
            for token in loop_messages:
                phrase.add_token(token)

        self.assertEqual("Cycled phrase detected", context.exception.message)
        self.assertEqual(('some normal output\n'
                          'first\n'
                          'second\n'
                          'third\n'
                          'first\n'
                          'second\n'
                          'third\n'
                          'first\n'
                          'second\n'
                          'third\n'
                          'first\n'
                          'second\n'
                          'third\n'
                          'first\n'), phrase.full)

    def test_loop_line_duplicated(self):
        """A repeated single line must raise LoopError with all copies kept."""
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "looo_line_duplicated.txt")
        loop_messages = loop_messages_file.read_text()

        phrase = Phrase()
        with self.assertRaises(LoopError):
            for token in loop_messages:
                phrase.add_token(token)

        self.assertEqual(('some normal output\n'
                          'duplicated\n'
                          'duplicated\n'
                          'duplicated\n'
                          'duplicated\n'
                          'duplicated\n'), phrase.full)

    def test_loop_in_one_line_false_case(self):
        """Duplication-like content must not raise LoopError."""
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line_false_case.txt").read_text(
            encoding="utf-8")
        phrase = Phrase()

        for token in loop_messages:
            phrase.add_token(token)

        # no errors

    def test_loop_in_one_line(self):
        """Loop inside one long line must raise LoopError and clean the line state."""
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line2.txt").read_text(encoding="utf-8")
        phrase = Phrase()
        with self.assertRaises(LoopError) as context:
            for token in loop_messages:
                phrase.add_token(token)

        self.maxDiff = None  # pylint: disable=invalid-name  # unittest attribute
        self.assertEqual(('ingframework.orm.jpa.persistenceunit.PersistenceUnitXmlElementAttribute;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlElementValue;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlNode;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlNodeList;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitNodeVisitor;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchema;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchemaElement;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchemaParser;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchemaWriter;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXMLLoader;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlElement;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlAttribute;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlElementAttribute;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlElementValue;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlNode;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlNodeList;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitNodeVisitor;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchema;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchemaElement;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchemaParser;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchemaWriter;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXMLLoader;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlElement;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlAttribute;\\nimport '
                          'org.spr'), context.exception.payload)
        self.assertEqual('Generated content appears to be a loop', context.exception.message)

        phrase.clean_current_line()

        self.assertEqual([], phrase.current_line)
        self.assertEqual({}, phrase.current_line_has_no_pair_tokens)
        self.assertEqual({}, phrase.duplicate_ranges)
        self.assertEqual({}, phrase.duplicate_ranges_reversed)
        self.assertEqual({}, phrase.duplicated_words)
        self.assertEqual({}, phrase.duplicates_islands)
        self.assertEqual({}, phrase.duplicates_islands_reversed)

    def test_no_loop_but_has_duplicates(self):
        """Success-case content with duplicates must not raise and renders visualizations."""
        # pylint: disable=line-too-long  # long literal expectations
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line3_success_case.txt").read_text(
            encoding="utf-8")

        phrase = Phrase()
        for token in loop_messages:
            # no error
            phrase.add_token(token)

        line = phrase.current_line[phrase.in_line_duplicates_detect_start_amount:]
        unused_tokens_as_end_of_phrase = visualize_tokens(line, phrase.current_line_has_no_pair_tokens)
        loop_part1 = visualize_reversed_ranges(line, phrase.duplicate_ranges_reversed)
        loop_part2 = visualize_ranges(line, phrase.duplicate_ranges)
        visual_islands = visualize_islands_reversed(line, phrase.duplicates_islands_reversed)
        self.maxDiff = None
        self.assertEqual(('pply(plugin = \\"io.spring.dependency-management\\")\\n\\ndependencies '
                          '{\\n    api(project(\\":idempotent-consumer\\"))\\n    '
                          'api(project(\\":storage-api-reactive\\"))\\n    '
                          'api(project(\\":postgres-jdbc\\"))\\n\\n    '
                          'implementation(\\"io.projectreactor:reactor-core\\")\\n\\n    '
                          'implementation(\\"org.postgresql:postgresql\\")\\n\\n    '
                          'implementation(\\"org.springframework.boot:spring-boot-starter-jooq\\")\\n    '
                          'implementation(\\"org.springframework.boot:spring-boot-a-tocon----re\\")\\n\\n    '
                          'implementation(\\"org.jooq:jooq\\")\\n    '
                          'implementation(\\"org.jooq:jooq-postgres-extensions\\")\\n}-- "-ew-text": '
                          '"plugins {\\n    ---------rar--\\n}\\napply(plugin = '
                          '\\"io.spring.dependency-management\\")\\n\\ndependencies {\\n    '
                          'api(project(\\":idempotent-consumer\\"))\\n    '
                          'api(project(\\":storage-api-reactive\\"))\\n    '
                          'api(project(\\":postgres-jdbc\\"))\\n\\n    '
                          'implementation(\\"io.projectreactor:reactor-core\\")\\n'),
                         loop_part1)
        self.assertEqual(('************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************ '
                          '-----    '
                          '********************************************************************************** '
                          '-------------------------  ++ && #######################         ~~~  '
                          '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'),
                         visual_islands)
        self.assertEqual(('p---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:----------\\----------------------------------------------------------------s---t-----------------------------------------------------------------------ut-c--figur----------------------------------------------------------------------------------p--------e-----i---\\-----", '
                          '-ne-_t---"- '
                          '-p--------------`java-libr--y`\\--\\-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'),
                         unused_tokens_as_end_of_phrase)
        self.assertEqual(loop_part1, loop_part2)

    def test_duplicated_parts_simple(self):
        """Visualize duplicated parts of a small looping sample."""
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line.txt")
        loop_messages = loop_messages_file.read_text()

        profiler = cProfile.Profile()
        profiler.enable()

        duplicate_ranges, duplicate_reversed_ranges, duplicates_islands_reversed, line, line_tokens = (
            get_islands_of_duplicated_parts(loop_messages))

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        unused_tokens_as_end_of_phrase = visualize_tokens(line, line_tokens)
        loop_part1 = visualize_reversed_ranges(line, duplicate_reversed_ranges)
        loop_part2 = visualize_ranges(line, duplicate_ranges)
        visual_islands = visualize_islands_reversed(line, duplicates_islands_reversed)

        self.assertEqual('----imoim---oimport\\nimport\\nimportiiiii', loop_part1)
        self.assertEqual('    -----   ****************************', visual_islands)
        self.assertEqual('ilhei-o--allo----------------------i----', unused_tokens_as_end_of_phrase)
        self.assertEqual(loop_part1, loop_part2)

    def test_duplicated_parts_repeated_consonants(self):
        """Visualize duplicated parts with repeated consonant tokens."""
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line4.txt")
        loop_messages = loop_messages_file.read_text()

        profiler = cProfile.Profile()
        profiler.enable()

        duplicate_ranges, duplicate_reversed_ranges, duplicates_islands_reversed, line, line_tokens = (
            get_islands_of_duplicated_parts(loop_messages))

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        unused_tokens_as_end_of_phrase = visualize_tokens(line, line_tokens)
        loop_part1 = visualize_reversed_ranges(line, duplicate_reversed_ranges)
        loop_part2 = visualize_ranges(line, duplicate_ranges)
        visual_islands = visualize_islands_reversed(line, duplicates_islands_reversed)

        self.assertEqual(
            'ribute;At;PersistenceUnitXmlAttribute;PersistenceUnitXmlAttribute;PersistenceUnitXmlAttribute;',
            loop_part1)
        self.assertEqual(loop_part1, loop_part2)
        self.assertEqual(
            '----------------------------------------------------------------------------------------------',
            visual_islands)
        self.assertEqual(
            'r------A-;------------------------------------------------------------------------------------',
            unused_tokens_as_end_of_phrase)

    def test_duplicated_parts_big(self):  # pylint: disable=too-many-locals  # prepares many visualizations at once
        """Visualize duplicated parts and the last-part islands of a big sample."""
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line2.txt").read_text(encoding="utf-8")
        loop_messages_expected_result = files(__package__).joinpath(TEST_RESOURCES,
                                                                    "loop_in_line2_result.txt").read_text(
            encoding="utf-8")

        loop_messages_expected_islands = files(__package__).joinpath(TEST_RESOURCES,
                                                                     "loop_in_line2_islands.txt").read_text(
            encoding="utf-8")

        profiler = cProfile.Profile()
        profiler.enable()

        duplicate_ranges, duplicate_reversed_ranges, duplicates_islands_reversed, line, line_tokens = (
            get_islands_of_duplicated_parts(loop_messages))

        start, end = get_last_part_border(line, duplicates_islands_reversed)
        last_part_islands_reversed = layout_last_island(line, start, end)
        start, end = get_last_part_border(line, last_part_islands_reversed)
        last_part_of_last_part_islands_reversed = layout_last_island(line, start, end)

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        _island_loop_part1 = visualize_reversed_ranges(line, last_part_of_last_part_islands_reversed)
        _visual_last_island = visualize_islands_reversed(line, last_part_of_last_part_islands_reversed)

        unused_tokens = visualize_tokens(line, line_tokens)
        loop_part1 = visualize_reversed_ranges(line, duplicate_reversed_ranges)
        loop_part2 = visualize_ranges(line, duplicate_ranges)
        visual_islands = visualize_islands_reversed(line, duplicates_islands_reversed)

        self.maxDiff = None
        self.assertEqual(loop_messages_expected_result, loop_part1)
        self.assertEqual(loop_part1, loop_part2)
        self.assertEqual(loop_messages_expected_islands, visual_islands)

    def test_duplicated_parts_case5(self):
        """Replay case 5 tokens and compare the full accumulated text."""
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line5_success_case.json").read_text(
            encoding="utf-8")

        phrase = Phrase()
        tokens = json.loads(loop_messages)
        full_expected = ""
        for token in tokens:
            full_expected += token
            phrase.add_token(token)

        # line = phrase.current_line[phrase.in_line_duplicates_detect_start_amount:]
        # loop_part1 = visualize_reversed_ranges(line, phrase.duplicate_ranges_reversed)
        # visual_islands = visualize_islands_reversed(line, phrase.duplicate_ranges_reversed)

        self.maxDiff = None
        self.assertEqual(full_expected, phrase.full)
        # self.assertEqual((''), loop_part1)
        # self.assertEqual((''), visual_islands)

    def test_duplicated_cycle_phrase_false_case(self):
        """A cycled-phrase false case must not raise LoopError."""
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_cycled_phrase_false_case.txt").read_text()

        phrase = Phrase()
        for token in loop_messages:
            phrase.add_token(token)

    def test_empty_tokens_are_ignored(self):
        """Empty decoded tokens add no text and a run of them is not a duplicated-token loop."""
        phrase = Phrase()
        for token in ["a", "", "", "b"]:
            phrase.add_token(token)
        self.assertEqual("ab", phrase.full)

        phrase = Phrase()
        for _ in range(DEFAULT_DUPLICATED_TOKENS_LIMIT * 5):
            phrase.add_token("")
        self.assertEqual("", phrase.full)

    def test_duplicated_tokens_limit_boundary(self):
        """The run limit trips on the token that reaches it, not before."""
        phrase = Phrase(strat_duplicates_detect_from=10_000)
        for _ in range(DEFAULT_DUPLICATED_TOKENS_LIMIT - 1):
            phrase.add_token("x")
        with self.assertRaises(LoopError) as context:
            phrase.add_token("x")
        self.assertEqual(f"Duplicated tokens (amount={DEFAULT_DUPLICATED_TOKENS_LIMIT})",
                         context.exception.message)

    def test_detection_can_be_disabled(self):
        """With detection off the phrase keeps the whole text and never raises."""
        phrase = Phrase(is_detect_looped_inference=False)
        text = "x" * (DEFAULT_DUPLICATED_TOKENS_LIMIT * 3) + "first\nsecond\n" * 12
        for token in text:
            phrase.add_token(token)
        self.assertEqual(text, phrase.full)

    def test_crlf_and_cr_complete_a_line(self):
        """Tokens ending with Windows or old-Mac newlines close the line as Unix ones do."""
        for new_line in ("\r\n", "\r"):
            with self.subTest(new_line=repr(new_line)):
                phrase = Phrase()
                phrase.add_token(f"first{new_line}")
                phrase.add_token(f"second{new_line}")
                self.assertEqual([f"first{new_line}", f"second{new_line}"], phrase.lines)
                self.assertEqual([], phrase.current_line)

    def test_cycled_phrase_with_crlf_lines(self):
        """A two-line cycle written with CRLF endings is detected as a cycled phrase."""
        phrase = Phrase()
        phrase.add_token("normal\r\n")
        with self.assertRaises(LoopError) as context:
            for _ in range(12):
                phrase.add_token("first\r\n")
                phrase.add_token("second\r\n")
        self.assertEqual("Cycled phrase detected", context.exception.message)

    def test_last_part_border_without_islands(self):
        """Without islands the border stays undefined."""
        self.assertEqual((None, None), get_last_part_border([], None))
        self.assertEqual((None, None), get_last_part_border(list("abcd"), None))


class CycledPhraseCase(unittest.TestCase):
    """Reference cases of the cycled multi-line phrase report."""

    def test_payload_is_one_cycle(self):
        """The reported cycled phrase is exactly one repeating unit, without extra lines."""
        phrase = Phrase()
        with self.assertRaises(LoopError) as context:
            for _ in range(12):
                phrase.add_token("first\n")
                phrase.add_token("second\n")
        self.assertEqual("Cycled phrase detected", context.exception.message)
        self.assertEqual(["first\n", "second\n"],
                         sorted(context.exception.payload.splitlines(keepends=True)))

    def test_cycle_of_three_lines(self):
        """A three-line cycle is detected and reported as its three lines."""
        phrase = Phrase(duplicated_lines_threshold=3)
        with self.assertRaises(LoopError) as context:
            for _ in range(4):
                for line in ("a\n", "b\n", "c\n"):
                    phrase.add_token(line)
        self.assertEqual("Cycled phrase detected", context.exception.message)
        self.assertEqual(["a\n", "b\n", "c\n"],
                         sorted(context.exception.payload.splitlines(keepends=True)))

    def test_long_cycle_detected_before_ten_repeats(self):
        """An 18-line cycle with blank lines must trip at 3 repeats, not after 10."""
        unit = ["Now I'm setting up the test configuration class that will handle the Spring beans\n", "\n",
                "Configuring the test setup...\n", "\n", "step one\n", "\n", "step two\n", "\n",
                "step three\n", "\n", "step four\n", "\n", "step five\n", "\n", "step six\n", "\n",
                "step seven\n", "\n"]
        phrase = Phrase()
        with self.assertRaises(LoopError) as context:
            for _ in range(5):
                for line in unit:
                    phrase.add_token(line)
        self.assertEqual("Cycled phrase detected", context.exception.message)


class LoopDetectorStateCase(unittest.TestCase):
    """Unit tests of the duplicate-island state helpers used by the loop detector."""

    def test_last_part_border_of_a_short_line(self):
        """The border search finds an island ending on the last token of a short line."""
        line = list("abcdabcd")
        islands = layout_last_island(line, 0, 7)
        self.assertEqual({7: 0}, islands)
        self.assertEqual((0, 7), get_last_part_border(line, islands))

    def test_merge_accepts_a_range_starting_at_zero(self):
        """A duplicated range starting at position 0 is merged, not skipped as falsy."""
        merged, end_positions = find_duplicated_with_longest_last(
            0, ("b",), [3], list("abab"), {3: 0}, {("a", "b", "a", "b"): {0}})
        self.assertEqual((("a", "b", "a", "b"), [3]), (merged, end_positions))


def get_islands_of_duplicated_parts(loop_messages: str) -> tuple[
    dict[int, int], dict[int, int], dict[int, int], list[str], dict[str, set[int]]]:
    """Run the pair detector over the message tokens and return the island state."""
    line = list[str]()
    line_tokens = dict[str, set[int]]()
    duplicate_reversed_ranges = dict[int, int]()
    duplicate_ranges = dict[int, int]()
    duplicated_words = dict[str, set[int]]()
    islands = dict[int, int]()
    duplicates_islands_reversed = dict[int, int]()
    for i, token in enumerate(loop_messages):
        add_token(token, line)
        add_check_duplicate_tokens(line_tokens, token, i)
        process_duplicate_pairs(token, line, line_tokens, duplicate_reversed_ranges, duplicate_ranges,
                                duplicated_words, islands, duplicates_islands_reversed)

    return duplicate_ranges, duplicate_reversed_ranges, duplicates_islands_reversed, line, line_tokens


if __name__ == '__main__':
    unittest.main()
