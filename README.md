# Agent RL Examples

Examples of training LLM agents with RL.

## Installation
For just trying out examples:
```bash
uv sync
```

For developing/contributing:
```bash
uv sync --extra dev
uv run pre-commit install
```

## Development

- Run formatters and linters via `uv run`:
```bash
uv run black .
uv run ruff check --fix .
uv run mypy
```

- Run all pre-commit checks:
```bash
uv run pre-commit run -a
```
