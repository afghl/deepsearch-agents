# Repository Guidelines

This repository contains a minimal multi-step research agent framework. Follow these guidelines when contributing.

## Coding standards
- Target **Python&nbsp;3.10** or newer.
- Follow standard PEP&nbsp;8 style (4 spaces per indent).
- Provide type hints for all new functions and methods.
- Document public classes and functions with docstrings.
- Use the logger from `deepsearch_agents.log` instead of `print`.

## Development workflow
1. Run **type checking** before committing:
   ```bash
   mypy src
   ```
2. If a `.pre-commit-config.yaml` file is present, run linting on the changed files:
   ```bash
   pre-commit run --files <changed files>
   ```
3. When adding dependencies, update both `pyproject.toml` and `uv.lock`.

## Pull requests
- Provide a concise summary of the changes.
- Include the results of any checks you executed.
