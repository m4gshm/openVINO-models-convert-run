plugins {
    `java-library`
}
apply(plugin = "io.spring.dependency-management")

dependencies {
    api(project(":idempotent-consumer"))
    api(project(":storage-api-reactive"))
    api(project(":postgres-jdbc"))

    implementation("io.projectreactor:reactor-core")

    implementation("org.postgresql:postgresql")

    implementation("org.springframework.boot:spring-boot-starter-jooq")
    implementation("org.springframework.boot:spring-boot-autoconfigure")

    implementation("org.jooq:jooq")
    implementation("org.jooq:jooq-postgres-extensions")
}

