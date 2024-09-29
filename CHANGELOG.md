# CHANGELOG

## 0.4.1

### Fix

- Fix API Connection Errors with Semaphore

## 0.4.0

### Added

- Added o1 support
- Added meta evaluation pipeline to Python package

### Fixed

- Change jinja template extensions

## 0.3.1

### Fixed

- Fixed prompts newlines
- Removed latest structured generation because faithfulness and usefulness dtos did not support it
- Fixed dataset loading in plot function

## 0.3.0

### Added

- Add flag to use the training dataset in meta evaluation.

## 0.2.1

### Fixed

- Add missing model register file.

## 0.2.0

### Added

- Register Fireworks Llama 3.1 8b and 70b prices with litellm to better support these models as evaluators.

### Changed

- Remove `instructor` package to better understand what is really sent to the LLM. All the LLM generations are simply done using `litellm.acompletion`.
- Removed black from justfile and dev dependencies as ruff plays the same role.

## 0.1.0

### Added

- Created `GroundedQAEvaluator` that evaluates four metrics per sample: answer relevancy, completeness, faithfulness, usefulness, negative rejection and positive acceptance.
- Created `MetaEvaluator` to evaluate evaluators on GroUSE unit tests.
