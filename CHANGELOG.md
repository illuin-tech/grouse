# CHANGELOG

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
