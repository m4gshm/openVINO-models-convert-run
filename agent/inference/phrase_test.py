import cProfile
import json
import pstats
import unittest
from importlib.resources import files

from agent.inference.loop_error import LoopError
from agent.inference.phrase import DUPLICATED_TOKENS_LIMIT, Phrase, \
    process_duplicate_pairs, visualize_ranges, add_token, \
    add_check_duplicate_tokens, visualize_reversed_ranges, visualize_tokens, visualize_islands_reversed, \
    get_last_part_border, layout_last_island

TEST_RESOURCES = "test_resources/phrase"


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
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line2.txt").read_text(encoding="utf-8")
        phrase = Phrase()
        with self.assertRaises(LoopError) as context:
            for token in loop_messages:
                phrase.add_token(token)

        self.assertEqual(('tion;\\nimport io.github.m4gshm.idempotent.consumer.MessageImpl;\\nimport '
                          'io.github.m4gshm.idempotent.consumer.storage.tables.InputMessages;\\nimport '
                          'io.r2dbc.postgresql.api.ClientOptions;\\nimport '
                          'io.r2dbc.postgresql.client.PoolConfig;\\nimport '
                          'io.r2dbc.postgresql.client.R2DBCClient;\\nimport '
                          'io.r2dbc.postgresql.codec.CodecRegistry;\\nimport '
                          'io.r2dbc.postgresql.pgsql96.PgSqlParameterSource;\\nimport '
                          'org.junit.jupiter.api.BeforeEach;\\nimport '
                          'org.junit.jupiter.api.Test;\\nimport '
                          'org.springframework.data.convert.JsonCodec;\\nimport '
                          'reactor.core.publisher.Flux;\\nimport reactor.core.publisher.Mono;\\nimport '
                          'reactor.core.scheduler.Schedulers;\\nimport '
                          'reactor.test.StepVerifier;\\nimport '
                          'r2dbc.jdbc.JdbcConnectionFactory;\\nimport '
                          'r2dbc.jdbc.JdbcConnectionPool;\\nimport '
                          'r2dbc.jdbc.JdbcDatabaseClient;\\nimport '
                          'r2dbc.jdbc.JdbcTransactionManager;\\nimport '
                          'r2dbc.jdbc.TransactionDefinition;\\nimport javax.sql.DataSource;\\nimport '
                          'org.springframework.jdbc.datasource.DriverManagerDataSource;\\nimport '
                          'org.springframework.jdbc.core.JdbcTemplate;\\nimport '
                          'org.springframework.transaction.reactive.TransactionalOperator;\\nimport '
                          'org.springframework.transaction.annotation.EnableTransactionManagement;\\nimport '
                          'org.springframework.transaction.annotation.Transactional;\\nimport '
                          'org.springframework.context.annotation.Configuration;\\nimport '
                          'org.springframework.stereotype.Component;\\nimport '
                          'org.springframework.boot.autoconfigure.SpringBootApplication;\\nimport '
                          'org.springframework.boot.SpringApplication;\\nimport '
                          'org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;\\nimport '
                          'org.springframework.boot.autoconfigure.security.servlet.SecurityServletAutoConfiguration;\\nimport '
                          'org.springframework.boot.autoconfigure.web.servlet.WebMvcAutoConfiguration;\\nimport '
                          'org.springframework.data.jpa.repository.JpaRepository;\\nimport '
                          'org.springframework.orm.jpa.LocalContainerEntityManagerFactoryBean;\\nimport '
                          'org.springframework.orm.jpa.vendor.DatabasePlatform;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitManager;\\nimport '
                          'org.springframework.orm.jpa.metamodel.MappingModelGeneratorProcessor;\\nimport '
                          'org.springframework.orm.jpa.support.EntityManagerCreator;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.DefaultPersistenceUnitManager;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitInfo;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitReader;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchema;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchemaElement;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchemaParser;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitSchemaWriter;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlReader;\\nimport '
                          'org.springframework.orm.jpa.persistenceunit.PersistenceUnitXmlWriter;\\nimport '
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
                          'org.springf'), context.exception.payload)
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

        self.assertEqual(('----re\\")\\n\\n    implementation(\\"org.jooq:jooq\\")\\n    '
                          'implementation(\\"org.jooq:jooq-postgres-extensions\\")\\n}-- "----text": '
                          '"plugins {\\n    ---------ra---\\n}\\nappl-(plugin '
                          '--\\"io.-pring.dependenc----nagement\\")\\n\\ndependenc-es {\\n    '
                          'api(project(\\":-dempotent-cons-me-\\"))\\n    '
                          'api(project(\\":storage-api-reacti-e\\"))\\n    '
                          'api(project(\\":postgres-----\\"))\\n\\n    '
                          'implementation(\\"io.projectreactor-reactor-core\\")\\n'),
                         loop_part1)
        self.assertEqual(('    '
                          '---------------------------------------------------------------------------------- '
                          '*************************  **    ~~~~~~~~~~~~~~~~~~~~~~~         &&   '
                          '&&&&&&&&& ++++++++  &&&&& ###############    ++++++++++++++++++++++++ '
                          '************************* ~~~~~~~~~~~~~~ -- ################################ '
                          '~~~ ++++++ -----------------------------------    '
                          '############################################## -----------------'), visual_islands)
        self.assertEqual((
            'figur------------------------------------------\\---------------------------------------p--------e-----i---\\-----", '
            '-new_t---"- -p--------------`java-libr-ry`\\--\\---p-y(-------= '
            '\\----sp-i-g-d--------y-man-------\\---------------ie------------------------id----t--------um-r\\------------------------s-------a---r-----ve-------------------------p--------jdbc\\-------------------------------p------r------:r----------------'),
            unused_tokens_as_end_of_phrase)
        self.assertEqual(loop_part1, loop_part2)

    def test_duplicated_parts_simple(self):
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line.txt")
        loop_messages = loop_messages_file.read_text(encoding="utf-8")

        profiler = cProfile.Profile()
        profiler.enable()

        duplicate_ranges, duplicate_reversed_ranges, duplicates_islands_reversed, line, line_tokens = get_islands_of_duplicated_parts(
            loop_messages)

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
        loop_messages_file = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line4.txt")
        loop_messages = loop_messages_file.read_text(encoding="utf-8")

        profiler = cProfile.Profile()
        profiler.enable()

        duplicate_ranges, duplicate_reversed_ranges, duplicates_islands_reversed, line, line_tokens = get_islands_of_duplicated_parts(
            loop_messages)

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

    def test_duplicated_parts_big(self):
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line2.txt").read_text(encoding="utf-8")
        loop_messages_expected_result = files(__package__).joinpath(TEST_RESOURCES,
                                                                    "loop_in_line2_result.txt").read_text(
            encoding="utf-8")

        loop_messages_expected_islands = files(__package__).joinpath(TEST_RESOURCES,
                                                                     "loop_in_line2_islands.txt").read_text(
            encoding="utf-8")

        profiler = cProfile.Profile()
        profiler.enable()

        duplicate_ranges, duplicate_reversed_ranges, duplicates_islands_reversed, line, line_tokens = get_islands_of_duplicated_parts(
            loop_messages)

        start, end = get_last_part_border(line, duplicates_islands_reversed)
        last_part_islands_reversed = layout_last_island(line, start, end)
        start, end = get_last_part_border(line, last_part_islands_reversed)
        last_part_of_last_part_islands_reversed = layout_last_island(line, start, end)

        # Format and display the statistics
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 bottlenecks

        island_loop_part1 = visualize_reversed_ranges(line, last_part_of_last_part_islands_reversed)
        visual_last_island = visualize_islands_reversed(line, last_part_of_last_part_islands_reversed)

        unused_tokens = visualize_tokens(line, line_tokens)
        loop_part1 = visualize_reversed_ranges(line, duplicate_reversed_ranges)
        loop_part2 = visualize_ranges(line, duplicate_ranges)
        visual_islands = visualize_islands_reversed(line, duplicates_islands_reversed)

        self.maxDiff = None
        self.assertEqual(loop_messages_expected_result, loop_part1)
        self.assertEqual(loop_part1, loop_part2)
        self.assertEqual(loop_messages_expected_islands, visual_islands)

        pass

    def test_duplicated_parts_case5(self):
        loop_messages = files(__package__).joinpath(TEST_RESOURCES, "loop_in_line5_success_case.json").read_text(
            encoding="utf-8")

        phrase = Phrase()
        tokens = json.loads(loop_messages)
        for token in tokens:
            phrase.add_token(token)

        line = phrase.current_line[phrase.in_line_duplicates_detect_start_amount:]
        loop_part1 = visualize_reversed_ranges(line, phrase.duplicate_ranges_reversed)
        visual_islands = visualize_islands_reversed(line, phrase.duplicate_ranges_reversed)

        self.maxDiff = None
        self.assertEqual((''), loop_part1)
        self.assertEqual((''), visual_islands)

        pass


def get_islands_of_duplicated_parts(loop_messages: str) -> tuple[
    dict[int, int], dict[int, int], dict[int, int], list[str], dict[str, set[int]]]:
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
