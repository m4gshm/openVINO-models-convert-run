"""Shared test fixtures for the parser test modules."""
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
