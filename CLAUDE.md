# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Commands

- Install: `uv sync` (Python 3.12, locked in `uv.lock`)
- Run the app: `uv run streamlit run app.py`
- There are no tests, linter, or formatter config. The `# fmt: skip` markers are there for a Black-style formatter.

## Things that span files

- To add an algorithm parameter, change three places: the field on the options dataclass, its name in that algorithm's `params` list in `core/algorithms/__init__.py`, and an `if "<name>" in algo.params` widget in `app.py`. The widget default in `app.py` overrides the dataclass default, so the two can differ (`margin`, for example).
- The `params` lists in `DRAWING` aren't read anywhere. `app.py` hard-codes the draw widgets and builds `draw_params` itself.
- `peaks` returns centers only, with empty `boxes`, `areas` and `labels`. The stars and boxes drawers iterate over boxes, so they draw nothing for peaks.
- `app.py` is a top-level Streamlit script that reruns on every widget change. Temp file paths and the upload's `file_id` are kept in `st.session_state` so they survive reruns. They're replaced, and the old files deleted, when the upload changes.
- Output is MJPG `.avi` written to a temp file. Media files (`*.mp4`, `*.mov`, `*.avi`, images) are gitignored.
