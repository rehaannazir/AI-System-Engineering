# Prompt Versioning

Versioned prompts for `researcher` and `summarizer`, with a registry to
render them, a validator to check LLM output against a required JSON
schema, and an evaluation harness that scores each prompt version over
repeated runs.

## Project Structure

| Path | Purpose |
|---|---|
| `prompts/` | Versioned prompt templates (`researcher.py`, `summarizer.py`) |
| `registry/` | `Prompt_Register` — fetch and render a given prompt version |
| `validators/` | `validate_output` — checks required fields are present in the JSON response |
| `evaluation/` | `v_evaluation.py` — runs each version N times against the LLM and scores pass/fail |
| `tests/` | Unit tests for the registry and validator |

## Evaluation Results — `summarizer` prompt

Each version was run **5 times** against `gemini-2.5-flash` on the same
input topic. A run counts as a **pass** if the response is valid JSON
containing both `summary` and `keywords`.

| Version | Passed | Total Runs | Success Rate |
|---------|:------:|:----------:|:------------:|
| v1.0    | 0      | 5          | 0%           |
| v1.1    | 3      | 5          | 60%          |
| v2.0    | 4      | 5          | 80%          |

### Notes

- **v1.0** — loose instructions with no explicit output schema; the model
  frequently returned prose instead of JSON, failing validation on every run.
- **v1.1** — added an explicit JSON example in the prompt, which lifted
  reliability to 60%, but occasional non-JSON or malformed responses remained.
- **v2.0** — added explicit constraints (word limit, tone, audience) plus a
  stricter "Return JSON only" instruction with an example, raising reliability
  to 80% and making it the current recommended version.

**Next steps:** investigate the remaining 20% failure mode on v2.0 (likely
occasional prose preamble before the JSON block) and consider adding
retry-with-repair or a stricter system instruction to push toward 100%.
