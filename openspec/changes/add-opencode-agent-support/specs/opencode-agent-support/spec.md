# Spec Delta

## Purpose

Detect OpenCode IDE agent sessions and apply the same tool-call argument fixing, context extraction, and loop detection that Veai IDE integration uses.

## ADDED Requirements

### Requirement: OpenCode agent detection

The system SHALL detect when a chat request originates from an OpenCode IDE agent by examining the first system message for the marker string "You are an AI agent running in OpenCode".

#### Scenario: Detect OpenCode agent session
- **WHEN** the first message in the conversation contains "You are an AI agent running in OpenCode"
- **THEN** `is_opencode_agent()` returns `True`

#### Scenario: Non-OpenCode session
- **WHEN** the first message does not contain the OpenCode marker
- **THEN** `is_opencode_agent()` returns `False`

### Requirement: OpenCode context extraction

The system SHALL extract user context (OS type, workdir) from the system prompt when an OpenCode agent session is detected, mirroring the Veai context extraction pattern.

#### Scenario: Extract OS and workdir from system prompt
- **WHEN** the system prompt contains `<project_information>` block with OS info and project path
- **THEN** `get_opencode_context()` returns a `UserContext` with populated `os_type`, `os_full`, and `workdir`

#### Scenario: Missing context block
- **WHEN** the system prompt does not contain a `<project_information>` block
- **THEN** `get_opencode_context()` returns a `UserContext` with default/empty values

### Requirement: OpenCode tool-call argument fixing

The system SHALL apply per-tool argument repair for OpenCode's tool schema (edit, glob, grep, read, shell, skill, subagent, webfetch, websearch, write, execute), normalizing field names and validating required arguments.

#### Scenario: Fix edit tool arguments
- **WHEN** an edit tool call has missing or misnamed fields (path, oldString, newString)
- **THEN** the fixer normalizes field names and validates required parameters

#### Scenario: Fix read tool arguments
- **WHEN** a read tool call has missing or misnamed fields (path, offset, limit)
- **THEN** the fixer normalizes field names and applies defaults

#### Scenario: Fix shell tool arguments
- **WHEN** a shell tool call has missing or misnamed fields (command, timeout, background)
- **THEN** the fixer normalizes field names and applies defaults

### Requirement: Loop detection for cycled tool calls

The system SHALL detect when the same tool call is repeated in a conversation and offer the user a choice to continue or interrupt.

#### Scenario: Detect repeated tool call
- **WHEN** the same tool name appears more than once in consecutive assistant messages
- **THEN** the system interrupts with a question offering "continue" or "interrupt" options

#### Scenario: User chooses continue
- **WHEN** the user selects "continue" after a loop detection interrupt
- **THEN** generation resumes with the same tool call

#### Scenario: User chooses interrupt
- **WHEN** the user selects "interrupt" after a loop detection interrupt
- **THEN** generation stops and returns control to the user

### Requirement: Engine wiring

The system SHALL wire OpenCode detection into `engine_rest_common.py` alongside the existing Veai detection, so both agents receive the same treatment without modifying the Veai code path.

#### Scenario: Both Veai and OpenCode agents work
- **WHEN** a request comes from either a Veai or OpenCode agent
- **THEN** the appropriate detection function triggers the correct context extraction and tool-fixing pipeline

#### Scenario: Unknown agent falls through gracefully
- **WHEN** a request does not match any known agent marker
- **THEN** the system uses default `UserContext` without errors

### Requirement: Loop detection in thinking blocks

The system SHALL detect when the model enters a thinking loop — repeating the same reasoning
without making progress — and interrupt generation before tokens are wasted.

#### Scenario: Detect thinking loop
- **WHEN** the assistant's thinking block contains the same or similar reasoning repeated
  more than three times without advancing toward a solution
- **THEN** the system interrupts generation and returns a warning message explaining the loop

#### Scenario: Thinking loop with tool calls
- **WHEN** the model is stuck in a thinking loop while attempting to make tool calls
- **THEN** the system detects the loop, interrupts generation, and offers the user a choice
  to continue or interrupt the current tool call sequence