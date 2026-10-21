"""Shared test fixtures for the tool call parser tests."""
from importlib.resources import files

TEST_RESOURCES = "test_resources"


class FakeTokenizer:  # pylint: disable=too-few-public-methods  # duck-typed stub
    """Tokenizer stub for parser tests: only what Parser.__init__ reads."""

    def get_eos_token(self) -> str:
        """The parser compares end tokens against this string."""
        return "<eos>"


GRADLE_PLUGINS_BLOCK = (
    'plugins {\n'
    '    `java-library`\n'
    '}\n'
    'apply(plugin = "io.spring.dependency-management")\n'
    '\n'
)

GRADLE_DEPENDENCIES_BLOCK = (
    'dependencies {\n'
    '    api(project(":idempotent-consumer"))\n'
    '    api(project(":storage-api-reactive"))\n'
    '    api(project(":postgres-jdbc"))\n'
    '\n'
    '    implementation("io.projectreactor:reactor-core")\n'
)

GRADLE_SPRING_BOOT_DEPENDENCIES = (
    '\n'
    '    implementation("org.postgresql:postgresql")\n'
    '\n'
    '    '
    'implementation("org.springframework.boot:spring-boot-starter-jooq")\n'
    '    '
    'implementation("org.springframework.boot:spring-boot-autoconfigure")\n'
    '\n'
    '    implementation("org.jooq:jooq")\n'
    '    '
    'implementation("org.jooq:jooq-postgres-extensions")\n'
)

GRADLE_SPRING_BOOT_DEPENDENCIES_NO_EOL = GRADLE_SPRING_BOOT_DEPENDENCIES[:-1]

GRADLE_REACTOR_AND_SPRING_BOOT_DEPENDENCIES = (
    'implementation("io.projectreactor:reactor-core")\n' + GRADLE_SPRING_BOOT_DEPENDENCIES
)

GRADLE_REACTOR_AND_SPRING_BOOT_DEPENDENCIES_NO_EOL = GRADLE_REACTOR_AND_SPRING_BOOT_DEPENDENCIES[:-1]

GRADLE_TESTCONTAINERS_DEPENDENCIES = (
    '\n'
    '    // Testcontainers dependencies\n'
    '    testImplementation("org.testcontainers:postgresql:1.19.7")\n'
    '    testImplementation("org.testcontainers:junit-jupiter:1.19.7")\n'
    '    testImplementation("org.testcontainers:containers:1.19.7")\n'
)

GRADLE_TESTCONTAINERS_JUNIT = (
    '\n'
    '    // --- Testcontainers Dependencies ---\n'
    '    testImplementation("org.testcontainers:junit-jupiter")\n'
    '    testImplementation("org.testcontainers:postgresql")\n'
    '    testImplementation("org.testcontainers:os")\n'
    '    testImplementation("org.junit.jupiter:junit-jupiter-api")\n'
    '    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")\n'
    '}'
)

GRADLE_TESTCONTAINERS_INTEGRATION = (
    '\n'
    '    // Testcontainers dependencies for integration testing\n'
    '    testImplementation("org.testcontainers:postgresql")\n'
    '    testImplementation("org.testcontainers:junit-jupiter")\n'
    '}'
)

GRADLE_TESTCONTAINERS_TEST = (
    '    // Testcontainers dependencies for integration tests\n'
    '    testImplementation("org.testcontainers:postgresql")\n'
    '    testImplementation("org.testcontainers:junit-jupiter")'
)

GRADLE_TESTCONTAINERS_DEVELOPMENT = (
    '    // TestContainers dependencies for integration tests\n'
    '    developmentImplementation("org.testcontainers:testcontainers")\n'
    '    developmentImplementation("org.testcontainers:junit-jupiter")\n'
    '    developmentImplementation("org.testcontainers:postgresql")'
)

GRADLE_BUILD_KTS_CONTENT = (
    GRADLE_PLUGINS_BLOCK + GRADLE_DEPENDENCIES_BLOCK + GRADLE_SPRING_BOOT_DEPENDENCIES + GRADLE_TESTCONTAINERS_JUNIT
)

GRADLE_BUILD_KTS_CONTENT_INTEGRATION = (
    GRADLE_PLUGINS_BLOCK + GRADLE_DEPENDENCIES_BLOCK + GRADLE_SPRING_BOOT_DEPENDENCIES
    + GRADLE_TESTCONTAINERS_INTEGRATION
)

GRADLE_DEPENDENCIES_AND_TESTCONTAINERS = (
    GRADLE_DEPENDENCIES_BLOCK + GRADLE_SPRING_BOOT_DEPENDENCIES + GRADLE_TESTCONTAINERS_DEPENDENCIES + '\n}\n'
)

GRADLE_BUILD_KTS_NEW_TEXT = (
    GRADLE_PLUGINS_BLOCK + GRADLE_DEPENDENCIES_BLOCK + GRADLE_SPRING_BOOT_DEPENDENCIES + GRADLE_TESTCONTAINERS_TEST
)

GRADLE_BUILD_KTS_OLD_TEXT = (
    GRADLE_PLUGINS_BLOCK + GRADLE_DEPENDENCIES_BLOCK + GRADLE_SPRING_BOOT_DEPENDENCIES_NO_EOL
)


def read_tool_call(package: str, resource: str) -> str:
    """Read a tool call expression from the test resources of a package."""
    return files(package).joinpath(TEST_RESOURCES, resource).read_text()


def parse_first_call(parser, state, package: str, resource: str):
    """Parse a resource tool call expression, returning its first call and the partial flag."""
    calls, partial = parser.parse_tool_calls(state, read_tool_call(package, resource))
    return calls[0], partial
