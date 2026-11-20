# Spec Delta

## Purpose

Publish OpenVINO IR model folders from the local `models/` directory to the `m4gshm` HuggingFace account, with a maintainable model list, idempotent re-runs (sync), and consistent README model cards for every published model.

## ADDED Requirements

### Requirement: Model list drives publishing
The publish script SHALL contain an explicit list of models to publish (9 Qwen2.5-Coder variants, all published to HF). Only these 9 models are in scope — other folders in `models/` (gemma-4, Qwen3.x, etc.) are explicitly excluded from publishing. The list SHALL be extendable by editing the script (or its config section) without code changes elsewhere.

#### Scenario: Listed models are published
- **WHEN** the script runs and each listed folder exists under `models/`
- **THEN** each model is uploaded to `m4gshm/<folder-name>` on HuggingFace

#### Scenario: New model added to the list
- **WHEN** a user adds a new folder name to the model list and re-runs the script
- **THEN** the new model is uploaded without modifying any other part of the script

#### Scenario: Missing folder
- **WHEN** a listed model folder does not exist under `models/`
- **THEN** the script reports the missing model and continues with the remaining models (non-zero exit only if nothing succeeded)

### Requirement: Re-runs sync existing repositories
Re-running the publish script SHALL be idempotent: models already on HuggingFace are updated in place, files whose content is unchanged are not re-uploaded, and no repository or file is duplicated or deleted.

#### Scenario: Model already published and unchanged
- **WHEN** the script runs for a model whose remote files all match local files
- **THEN** no files are re-uploaded and the script reports the model as up to date

#### Scenario: Model updated locally
- **WHEN** the script runs for a model with changed or new local files
- **THEN** changed/new files are uploaded, other remote files are left intact

### Requirement: Model card generation
Before uploading a model, the script SHALL ensure the model folder contains a `README.md` model card generated from a shared template with YAML front matter (`license`, `base_model`, tags, and quantization info derived from the folder name/config), prepared on the pattern of the models already published at https://huggingface.co/m4gshm. An existing hand-written README SHALL NOT be overwritten without an explicit opt-in.

#### Scenario: Model without README
- **WHEN** the script processes a model folder that has no `README.md`
- **THEN** a model card is generated from the template and included in the upload

#### Scenario: Model with existing README
- **WHEN** the script processes a model folder that already has a `README.md`
- **THEN** that README is kept as is and uploaded unchanged

### Requirement: Repository creation and authentication
The script SHALL create the target HuggingFace repository on first publish (public, model type) and SHALL fail with a clear message when authentication (`HF_TOKEN` or cached login) is missing or lacks write access.

#### Scenario: First publish of a model
- **WHEN** `m4gshm/<model-name>` does not exist yet
- **THEN** the script creates a public model repo and uploads the folder

#### Scenario: No credentials
- **WHEN** no HuggingFace token is available
- **THEN** the script exits with an explanatory error before uploading anything
