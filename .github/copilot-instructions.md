# Copilot Instructions for light-cv

## Project Overview
**light-cv** is a light blob detection and tracking system for video analysis using OpenCV. It provides multiple detection algorithms (Otsu thresholding, percentile-based, peak detection, and fixed threshold) to identify and visualize light sources in video frames.

## Architecture

### Core Structure
- **`app.py`**: Streamlit web UI for real-time video processing and downloading results
- **`core/algorithms/percentile.py`**: Percentile-based thresholding algorithm
- **`core/algorithms/otsu.py`**: Otsu's automatic thresholding algorithm
- **`core/algorithms/peaks.py`**: Local peak detection algorithm
- **`core/algorithms/fixed.py`**: Fixed threshold with margin algorithm
- **`core/utils.py`**: Shared visualization functions (stars, lines, boxes)
- **`core/settings.py`**: Centralized algorithm registry and configuration

### Data Flow
Each algorithm module exposes two key functions:
1. **`<algorithm>_thresholding(frame, **params) -> MatLike`**: Core thresholding function returning binary mask
2. **`process_frame(frame, options) -> (thresh, contours, centers, metadata)`**: Full pipeline with detection

Processing pipeline:
1. **Input**: Frame from video (BGR numpy array)
2. **Preprocessing**: Convert BGR → grayscale → Gaussian blur
3. **Thresholding**: Apply algorithm-specific method (returns MatLike binary mask)
4. **Morphological cleanup**: OPEN (remove noise) → CLOSE (fill holes)
5. **Detection**: Find contours and filter by area
6. **Metadata extraction**: Centers, boxes, areas, labels
7. **Visualization**: Draw results on overlay frame (in app.py)
8. **Output**: Write to video file

## Key Patterns

### Configuration via Dataclasses
Each algorithm uses a `@dataclass` for configuration stored in `core/settings.py`:
- **`PercentileOptions`**: `percentile` (0-100), `blur_size`, `clean_size`, `min_area`, `max_area`
- **`OtsuOptions`**: `blur_size`, `clean_size`, `min_area`, `max_area`
- **`PeaksOptions`**: `margin` (0-1), `blur_size`, `clean_size`, `peak_size`
- **`FixedOptions`**: `margin` (0-1), `blur_size`, `clean_size`, `min_area`, `max_area`

Algorithms registered in `ALGORITHMS` dict with `AlgorithmConfig` containing name, options class, process_frame function, and param list.

### Visualization Functions
Located in `core/utils.py`:
- `draw_stars()`: Star-shaped markers at blob centers (using geometric math)
- `draw_lines()`: Connect closest blob pairs (graph traversal with degree limit)
- `draw_boxes()`: Bounding rectangles with labels
- All functions modify overlay images in-place, not originals

### Error Handling
Video I/O operations use `RuntimeError` for fatal failures:
```python
if not vid.isOpened():
    raise RuntimeError("Failed to read input video")
```
Add similar guards when extending video operations.

## Development Workflows

### Adding a New Detection Algorithm
1. Create `core/algorithms/new_algorithm.py` following pattern in `core/algorithms/percentile.py`
2. Define `@dataclass` options class (e.g., `NewOptions`) with algorithm parameters
3. Implement `new_thresholding(frame: MatLike, **params) -> MatLike` returning binary mask
4. Implement `process_frame(frame: MatLike, options: Optional[NewOptions] = None)` returning tuple
5. Register in `core/settings.py` ALGORITHMS dict with `AlgorithmConfig`
6. Algorithm automatically appears in Streamlit UI dropdown with dynamic parameters

### Extending Streamlit UI
The app dynamically generates parameter controls from algorithm configuration:
- Get `algorithm = ALGORITHMS[selection]` from registry
- Create options instance: `options = algorithm.options(**params)`
- Process frame: `thresh, contours, centers, metadata = algorithm.process_frame(frame, options)`
- Apply visualizations (stars, boxes, lines) to overlay
- Write to output video

No UI code changes needed - just register new algorithm in settings.py

### Running the Application
```bash
# Streamlit app (primary interface)
streamlit run app.py

# Using uv (recommended)
uv run streamlit run app.py
```

## Important Notes

### Video Codec & FPS
- Output codec: XVID (`.avi` format)
- Default FPS: 30 if source FPS is invalid
- Use `cv2.VideoWriter_fourcc(*"XVID")` consistently

### Imports Pattern
- Use absolute imports from core package: `from core.settings import ALGORITHMS`, `from core.utils import draw_stars`
- Algorithm modules import MatLike: `from cv2.typing import MatLike`
- Settings imports algorithms: `from .algorithms import percentile, otsu, peaks, fixed`
- Use `@dataclass` decorator and `Optional` for type hints

### Contour Processing
- Always use `cv2.RETR_EXTERNAL` (outer contours only)
- Filter by area to eliminate noise and false positives
- Centers calculated as: `(x + w/2, y + h/2)` from bounding rect

### Code Style
- **Docstrings**: Always add a blank line after docstrings (before code starts)
- **Type Hints**: Use `MatLike` from `cv2.typing` for thresholding functions' return type
- **Formatting**: Use `# fmt: skip` to prevent formatter from breaking long type hint lines
- **Dataclasses**: Use `@dataclass` for all options/configuration classes
- Follow PEP 8 conventions for Python code
- Use descriptive variable names matching existing patterns (`thresh`, `blur`, `gray`)

## Documentation Maintenance

**Important**: Whenever code changes, algorithms are modified, or new features are added, **update `docs/ARCHITECTURE.md`** to reflect the changes. Keep the documentation synchronized with the codebase to maintain accuracy for future development and AI agent context. This includes:
- Algorithm parameter changes
- Processing pipeline modifications
- New visualization functions
- UI workflow changes
- Integration point updates
