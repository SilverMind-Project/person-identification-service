# Python environment

- Run Python only through `uv run --frozen ...` from the repository root, or a `make` target.
  Never system Python, bare `python`, or `pip install`.
- `uv.lock` changes only when you re-lock on purpose (`uv lock`); revert any other change.
- Format only the files you touched (`uv run --frozen ruff format <files>`).
