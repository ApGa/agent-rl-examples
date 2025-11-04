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

## Run an experiment.
E.g., To run the number_search example with GRPO on a single node (tested with 4 A100s with 80GB VRAM each):

First, change the filepaths and allocation_mode in /home/azureuser/source/agent-rl-examples/src/agent_rl/examples/number_search/config_grpo.yaml as appropriate.
```bash
export WANDB_API_KEY=...
uv run python -m areal.launcher.local src/agent_rl/examples/number_search/train_grpo.py --config src/agent_rl/examples/number_search/config_grpo.yaml
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
