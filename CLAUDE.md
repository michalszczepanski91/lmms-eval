# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

lmms-eval is an evaluation framework for large multimodal models (image, video, audio). A deeper agent-oriented guide lives in [skills/lmms-eval-guide/SKILL.md](skills/lmms-eval-guide/SKILL.md) with detailed references under `skills/lmms-eval-guide/references/`; the human guides are [docs/guides/model_guide.md](docs/guides/model_guide.md) and [docs/guides/task_guide.md](docs/guides/task_guide.md).

## Commands

Always use `uv`, never `pip`.

```bash
uv sync                                   # install
uv run pre-commit run --all-files         # lint + format (ruff check --fix, ruff format; line length 240)

# Tests (CPU-only, no downloads/API keys needed for the default suite)
uv run python -m pytest test/ -q --ignore=test/eval/test_usage_metrics.py
uv run python -m pytest test/eval/test_protocol.py -q                  # single file
uv run python -m pytest test/eval/test_protocol.py::test_name -q       # single test
uv run python -m pytest test/ -m "not gpu and not api and not slow" -q # skip marked tests

# Smoke eval (validate any task/model change this way with --limit)
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct --tasks mme --batch_size 1 --limit 8
lmms-eval tasks list        # browse tasks;  lmms-eval models --aliases  to list model backends
```

CI ("Hermetic CPU Contracts", [.github/workflows/hermetic-cpu-contracts.yml](.github/workflows/hermetic-cpu-contracts.yml)) runs a fixed subset of test files with `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=""`, so those tests must not touch the network or GPU.

## Architecture

**CLI entry** — `lmms-eval` → [lmms_eval/cli/dispatch.py](lmms_eval/cli/dispatch.py) routes subcommands (`eval`, `tasks`, `models`, `ui`, `serve`, `mcp`, `power`, `version`). Legacy flat invocation (`--model X --tasks Y` with no subcommand) is routed to `eval`, whose argument parsing and orchestration live in [lmms_eval/__main__.py](lmms_eval/__main__.py) (`parse_eval_args`, `cli_evaluate`). `--config` accepts a YAML run config with CLI overrides.

**Pipeline** — [lmms_eval/evaluator.py](lmms_eval/evaluator.py): `simple_evaluate` instantiates the model, builds the task dict via `TaskManager` ([lmms_eval/tasks/__init__.py](lmms_eval/tasks/__init__.py)), then `evaluate` constructs `Instance` requests ([lmms_eval/api/instance.py](lmms_eval/api/instance.py)), dispatches them to the model by `output_type` (`generate_until`, `loglikelihood`, `generate_until_multi_round`, plus an agentic multi-round loop), applies filters, runs each task's `process_results`, and aggregates metrics. Response caching is in `lmms_eval/caching/`; LLM-as-judge scoring in `lmms_eval/llm_judge/`.

**Two model families — this distinction drives everything.** Every model subclasses `lmms` ([lmms_eval/api/model.py](lmms_eval/api/model.py)) and sets `is_simple`:
- **Chat** (`models/chat/`, `is_simple = False`, preferred for new work): tasks are loaded as `ConfigurableMessagesTask`; `Instance.args` is `(doc_to_messages, gen_kwargs, doc_id, task, split)`, and messages are structured `ChatMessages` ([lmms_eval/protocol.py](lmms_eval/protocol.py)) with typed text/image/video/audio content.
- **Simple** (`models/simple/`, legacy, `is_simple = True`): tasks are loaded as `ConfigurableTask`; `Instance.args` is `(contexts, gen_kwargs, doc_to_visual, doc_id, task, split)` with `<image>` placeholders in text.

The evaluator picks the task class from `lm.is_simple`, so the same task YAML is consumed differently depending on the model. Models fetch the doc themselves via `self.task_dict[task][split][doc_id]`.

**Model registration** — [lmms_eval/models/__init__.py](lmms_eval/models/__init__.py) holds `AVAILABLE_SIMPLE_MODELS`, `AVAILABLE_CHAT_TEMPLATE_MODELS` (name → class name) and `MODEL_ALIASES`; these become `ModelManifest`s in `ModelRegistryV2` ([lmms_eval/models/registry_v2.py](lmms_eval/models/registry_v2.py)). A model id can have both a chat and simple implementation; chat wins unless forced simple. External plugins register via the `lmms_eval.models` entry-point group. Model-specific dependencies must be imported with `optional_import` from [lmms_eval/imports.py](lmms_eval/imports.py) so the registry imports cleanly without them.

**Tasks** — each benchmark is a directory under `lmms_eval/tasks/<name>/` with YAML configs (discovered automatically by `TaskManager`) and a `utils.py`. YAML references Python via `!function utils.fn` (`doc_to_visual`, `doc_to_text`, `doc_to_messages`, `doc_to_target`, `process_results`), supports `include:` for shared templates (e.g. `_default_template_yaml`), and `group` configs aggregate subtasks. Shared helpers (MCQ extraction, media resolution, math verification, video loading, GPT-judge utils) live in `lmms_eval/tasks/_task_utils/`. `lmms_eval_specific_kwargs` in YAML carries per-model prompt variants.

**Other surfaces** — `lmms_eval/entrypoints/` is the HTTP eval server (async job queue for training-time eval: `/evaluate`, `/jobs/{id}`, `/queue`); `lmms_eval/tui/` the web/terminal UI; `lmms_eval/mcp/` an MCP server.

## Tests

Tests mirror pipeline layers (see [test/README.md](test/README.md)): `test/cli/`, `test/models/` (registry resolution), `test/eval/` (task loading, request construction contract, protocol, evaluator, token counts), `test/cache/`. `test/eval/prompt_stability/` compares prompts for classic benchmarks against golden snapshots — changes to prompt formatting for those tasks will fail these tests intentionally.

## Contribution conventions

- Conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `style:`, `ci:`.
- PRs follow [.github/pull_request_template.md](.github/pull_request_template.md). PRs that add a task YAML under `lmms_eval/tasks/` or a model file under `lmms_eval/models/` are checked by CI for real end-to-end evidence: `E2E status: PASS`, the exact `python -m lmms_eval`/`lmms-eval` command, model/backend, split with `N=<count>`, hardware, result, evidence, and the checked attestation. Mock-only tests don't count; without a real run the PR must stay in draft.
