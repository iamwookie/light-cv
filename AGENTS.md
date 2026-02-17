# Project Guidelines

## Code Style
- Python with type hints; algorithm implementations use `cv2.typing.MatLike` and `dataclass` option objects. See [core/algorithms/otsu.py](core/algorithms/otsu.py) and [core/algorithms/percentile.py](core/algorithms/percentile.py).
- Each algorithm exposes `process_frame(frame, options)` and returns `AlgorithmMetadata` with `centers`, `boxes`, `areas`, and `labels`. See [core/algorithms/__init__.py](core/algorithms/__init__.py).
- Drawing helpers accept `AlgorithmMetadata` plus an options dataclass. See [core/draw/stars.py](core/draw/stars.py) and [core/draw/boxes.py](core/draw/boxes.py).

## Architecture
- Streamlit app drives UI, video upload, per-frame processing, and output download. See [app.py](app.py).
- Algorithm registry lives in `ALGORITHMS`, mapping names to `AlgorithmConfig` with options + processing function returning `AlgorithmMetadata`. See [core/algorithms/__init__.py](core/algorithms/__init__.py).
- Drawing helpers are split into `core/draw` modules and exposed via `DRAWING` using `DrawConfig`, taking `AlgorithmMetadata`. See [core/draw/__init__.py](core/draw/__init__.py).

## Build and Test
- No build/test commands are documented in the repo. Dependencies are listed in [pyproject.toml](pyproject.toml).

## Project Conventions
- UI parameters are driven by `algo.params` and conditionally surfaced in Streamlit. See [app.py](app.py) and [core/algorithms/__init__.py](core/algorithms/__init__.py).
- Algorithm and drawing registries share the same shape: config dataclass + `options()` factory + `params` list for UI. See [core/algorithms/__init__.py](core/algorithms/__init__.py) and [core/draw/__init__.py](core/draw/__init__.py).
- Algorithm options are dataclasses with defaults; `options()` constructs them via `AlgorithmConfig`. See [core/algorithms/__init__.py](core/algorithms/__init__.py).
- Drawing options are dataclasses; Streamlit builds per-feature options and passes them via `DRAWING["..."]`. See [app.py](app.py) and [core/draw/__init__.py](core/draw/__init__.py).

## Integration Points
- External dependencies: OpenCV for image processing and Streamlit for the UI. See [pyproject.toml](pyproject.toml) and [app.py](app.py).
- Video I/O uses temporary files and OpenCV `VideoCapture`/`VideoWriter`. See [app.py](app.py).

## Security
- The app accepts arbitrary video uploads and processes them locally via temp files. See [app.py](app.py).
