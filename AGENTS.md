# Repository Guidelines

## Project Structure & Module Organization
- Core Python package lives in `hmr4d/` (datasets, networks, configs, utils, datamodule). Prefer adding new code under the closest existing subpackage (e.g., `hmr4d/utils/geo` for geometry helpers).
- Experiment and training entry points are in `tools/train.py` and Hydra configs under `hmr4d/configs/` (`global/`, `exp/`, `data/`, `hydra/`).
- Demos and scripts for end users are in `tools/demo/` and `tools/video/`; keep them lightweight and CLI-oriented.
- Documentation lives in `docs/`, third‑party code in `third-party/`, and expected runtime folders are `inputs/` and `outputs/` (not tracked by git).

## Build, Test, and Development Commands
- Create the environment and install in editable mode:
  - `conda create -n gvhmr python=3.10 && conda activate gvhmr`
  - `pip install -r requirements.txt && pip install -e .`
- Run a quick demo:
  - `python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s`
- Train / evaluate:
  - `python tools/train.py exp=gvhmr/mixed/mixed`
  - `python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=...`

## Coding Style & Naming Conventions
- Python only; use 4-space indentation and keep code Black-compatible (see `pyproject.toml`, line length 120). Run `black .` before large refactors.
- Use `snake_case` for functions/modules, `CamelCase` for classes, and descriptive names for configs (e.g., `gvhmr/mixed/mixed`).
- Prefer type hints where they already exist and follow existing logging and Hydra patterns (e.g., `print_cfg`, `parse_args_to_cfg`).

## Testing Guidelines
- There is no formal pytest suite; use lightweight sanity scripts:
  - `python tools/unitest/run_dataset.py` for dataset/loader checks.
  - `python tools/unitest/make_hydra_cfg.py` to validate configuration parsing.
- For changes affecting training or evaluation, run a short demo or a small subset experiment and mention the command you used in the PR.

## Commit & Pull Request Guidelines
- Write concise, imperative commit titles; optional prefixes like `fix:` or `update:` are acceptable (e.g., `fix: install instructions`).
- For pull requests, include:
  - A brief summary of the change and motivation.
  - Any new commands, config options, or dependencies.
  - Sample logs, screenshots, or metrics if you modify training/evaluation behavior.

## Agent-Specific Instructions
- Keep edits minimal and localized; avoid reformatting or renaming outside the intended change.
- Reuse existing utilities and configuration mechanisms instead of introducing parallel abstractions.
- Do not add new dependencies without a clear justification in `requirements.txt` and the PR description.
- When adding explanations, comments, or documentation, prefer writing the main narrative in Simplified Chinese; keep original English terms alongside Chinese where it improves clarity.
