# Architecture Documentation

## System Overview

light-cv is a modular blob detection system with a **Streamlit web UI** (`app.py`) as the primary interface. The system abstracts multiple detection algorithms that users can leverage through the web interface.

```
┌─────────────────────────────────────────────────────────────────┐
│                   app.py (Streamlit Web UI)                      │
│                 (Primary User Interface)                         │
└────────────────────┬─────────────────────────────────────────────┘
                     │
         ┌───────────▼───────────────────────┐
         │  core/settings.py                 │
         │  (Algorithm Registry)             │
         └───────────▼───────────────────────┘
                     │
         ┌───────────▼────────────────────────────┐
         │  core/algorithms/                      │
         │  (Detection Modules)                   │
         │                                        │
         ├─ percentile.py ─────────────────────┐
         ├─ otsu.py ────────────────────────────┤ (thresholding functions)
         ├─ peaks.py ────────────────────────────┤ (process_frame)
         └─ fixed.py ────────────────────────────┘
                     │
         ┌───────────▼────────────────────────────┐
         │   core/utils.py                        │
         │   (Visualization)                      │
         │                                        │
         ├─ draw_stars()                         │
         ├─ draw_lines()                         │
         └─ draw_boxes()                         │
                     │
         ┌───────────▼────────────────────────────┐
         │ OpenCV (Video I/O)                     │
         └────────────────────────────────────────┘
```

## Module Reference

### Algorithm Registry and Configuration

**`core/settings.py`** provides the centralized `AlgorithmConfig` dataclass:

```python
@dataclass
class AlgorithmConfig:
    name: str                                      # Display name
    options_class: Type[Any]                       # Dataclass for parameters
    process_frame: Callable                        # Function reference
    params: List[str]                              # Parameter names for UI
```

All algorithms registered in `ALGORITHMS` dict, enabling:

- Dynamic UI generation in Streamlit (no code changes needed to add algorithms)
- Unified configuration management
- Extensible architecture

### Detection Algorithms

Each algorithm module in `core/algorithms/` contains:

1. **`<Algorithm>Options` dataclass**: Algorithm parameters (with defaults)
2. **`<algorithm>_thresholding()` function**: Core thresholding returning `MatLike`
3. **`process_frame()` function**: Full pipeline returning (thresh, contours, centers, metadata)

#### `core/algorithms/percentile.py` - Percentile Thresholding

- **Options**: `PercentileOptions` dataclass
- **Method**: Uses numpy percentile of grayscale values
- **Key Parameter**: `percentile` (0-100, higher = fewer detections)
- **Use Case**: Adaptive thresholding based on frame brightness distribution
- **Tuning**: Increase percentile for sparse/bright blobs, decrease for dense/dim

#### `core/algorithms/otsu.py` - Otsu's Method

- **Options**: `OtsuOptions` dataclass
- **Method**: OpenCV's automatic OTSU thresholding
- **Key Parameters**: `min_area`, `max_area`, `blur_size`, `clean_size`
- **Use Case**: Automatic threshold calculation without manual tuning
- **Tuning**: max_area typically set to 1/6 of frame area

#### `core/algorithms/peaks.py` - Local Peak Detection

- **Options**: `PeaksOptions` dataclass
- **Method**: Finds local intensity maxima using distance transform
- **Key Parameters**: `margin` (0-1), `peak_size`, `blur_size`, `clean_size`
- **Use Case**: Detects isolated bright spots (point sources)
- **Tuning**: Lower margin (0.5) for dim sources, higher (0.9) for bright

#### `core/algorithms/fixed.py` - Fixed Margin Threshold

- **Options**: `FixedOptions` dataclass
- **Method**: Threshold = MAX_INTENSITY × MARGIN
- **Key Parameters**: `margin` (0-1), `blur_size`, `clean_size`, `min_area`, `max_area`
- **Use Case**: Simple, deterministic thresholding for consistent lighting
- **Tuning**: margin is the most critical parameter (0.5-0.9 typical)

### Processing Pipeline

Each algorithm follows this standardized two-layer architecture:

**Layer 1: Thresholding** (`<algorithm>_thresholding()` function)

```python
# 1. COLOR CONVERSION
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 2. NOISE REDUCTION
blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

# 3. THRESHOLDING (algorithm-specific)
# Returns MatLike (binary mask)

# 4. MORPHOLOGICAL CLEANUP
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_size, clean_size))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove noise
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1) # Fill holes

return thresh  # Type: MatLike
```

**Layer 2: Detection & Metadata** (`process_frame()` function)

```python
# 5. CONTOUR DETECTION
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 6. FILTERING & METADATA EXTRACTION
for contour in contours:
    area = cv2.contourArea(contour)
    if options.min_area <= area <= options.max_area:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        # Collect centers, boxes, areas, labels

return thresh, contours, centers, metadata
```

### Visualization Functions (`core/utils.py`)

#### `draw_stars(img, centers, boxes, *, labels=None, ...)`

- Draws geometric 10-pointed star shapes at blob centers
- Inner points: 38.2% of outer radius (golden ratio)
- Efficient for identifying detection centers
- Labels placed above (if room) or below box

#### `draw_lines(img, centers, degree=2, ...)`

- Connects closest blob pairs with lines
- Graph-based: each blob connects to max `degree` neighbors (default: 2)
- Useful for tracking blob proximity/relationships
- Algorithm: Sort all pairs by distance, assign greedily respecting degree limit

#### `draw_boxes(img, boxes, *, labels=None, ...)`

- Axis-aligned bounding rectangles
- Simple, clear representation
- Each box labeled with detection info (index, area, etc.)

**Important**: All drawing functions operate on overlay image, not original frame.

## Video I/O Details

### Input/Output Pattern

```python
# INPUT
cap = cv2.VideoCapture(INPUT_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Fallback to 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# OUTPUT
fourcc = cv2.VideoWriter.fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
```

### FPS Handling

- **Default Fallback**: 30 FPS if `cap.get(cv2.CAP_PROP_FPS)` returns 0 or invalid
- **Critical**: Use same FPS for input and output to maintain temporal consistency
- **Why XVID**: Widely supported, good compression, reliable encoding

### Error Guards

All video I/O operations must guard against failures:

```python
if not cap.isOpened():
    raise RuntimeError("Failed to read input video")

if not ok:  # frame read failed
    raise RuntimeError("Failed to read frame")

if not out.isOpened():
    raise RuntimeError("Failed to create VideoWriter")
```

## Streamlit Integration (`app.py`)

### Flow

1. **Sidebar Configuration**:
    - Algorithm selector dropdown (`ALGORITHMS.keys()`)
    - Dynamic parameter controls generated from selected algorithm's `params` list
    - Visualization toggles (draw_stars, draw_boxes, draw_lines)

2. **Upload & Process**:
    - User selects video file (mp4, mov, avi supported)
    - File saved to temp location with original suffix
    - On "LETS COOK! 🍳" button click:
        - Extract video metadata (FPS, dimensions, frame count)
        - Create output VideoWriter for `.avi` format
        - For each frame:
            - `algorithm = ALGORITHMS[selection]`
            - `options = algorithm.options(**params)`
            - `thresh, contours, centers, metadata = algorithm.process_frame(frame, options)`
            - Apply selected visualizations (stars, boxes, lines) to overlay
            - Write to output with progress indicator

3. **Download**:
    - User downloads processed video as `.avi`

### Dynamic UI Generation

UI parameters automatically generated from algorithm configuration:

```python
algorithm = ALGORITHMS[selection]  # Get from registry
params = {}
# Dynamically create UI controls based on algorithm.params list
if "percentile" in algorithm.params:
    params["percentile"] = st.number_input(...)
if "blur_size" in algorithm.params:
    params["blur_size"] = st.slider(...)
# ... etc for all params
```

Adding new algorithms requires **zero UI changes** - just register in `core/settings.py`.

## Import Strategy

Project uses **package-based imports** with type hints:

```python
# In app.py
from core.settings import ALGORITHMS
from core.utils import draw_stars, draw_boxes, draw_lines

# In core/settings.py
from .algorithms import percentile, otsu, peaks, fixed
from cv2.typing import MatLike

# In algorithm modules (core/algorithms/*.py)
from cv2.typing import MatLike
from dataclasses import dataclass
from typing import Optional

def <algorithm>_thresholding(frame: MatLike, **params) -> MatLike:  # fmt: skip
    ...

def process_frame(frame: MatLike, options: Optional[<Options>] = None):
    ...
```

**Key conventions**:

- Absolute imports from `core` package in app.py and settings.py
- Relative imports within `core` package (e.g., `from .algorithms import ...`)
- Use `MatLike` from `cv2.typing` for OpenCV matrix types in function signatures
- Use `Optional` for dataclass parameters
- Use `# fmt: skip` on long type hint lines

## Data Types & Conventions

### Type Hints

- **Function Parameters**: Use `MatLike` for frames (OpenCV images)
- **Function Returns**: Thresholding functions return `MatLike` (binary mask)
- **Optional Parameters**: Use `Optional[<Type>]` for optional arguments
- **Dataclass Options**: All algorithm configurations are `@dataclass` types

### Coordinates

- **(x, y)**: Pixel position, origin top-left, increasing downward/rightward
- **Centers**: `(cx, cy)` = `(x + w//2, y + h//2)` from bounding rect
- **Boxes**: `(x, y, w, h)` tuple from `cv2.boundingRect(contour)`

### Image Formats

- **Frame**: BGR uint8 numpy array from `cv2.read()` (MatLike type)
- **Gray**: Single-channel uint8 from `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`
- **Binary/Thresh**: Single-channel uint8, values 0 or 255 from thresholding (MatLike type)
- **Overlay**: Copy of original frame for non-destructive drawing

### Contour/Area

- **Contour**: Numpy array of boundary points from `cv2.findContours()`
- **Area**: Float from `cv2.contourArea(contour)`, in pixels²
- **Filtering**: Always filter by area using `min_area` and `max_area` from options
- **Typical range**: 2-86400 pixels²

## Parameter Tuning Reference

| Parameter  | Range      | Effect                              | Algorithms              |
| ---------- | ---------- | ----------------------------------- | ----------------------- |
| percentile | 0-100      | Higher % = fewer detections         | percentile              |
| margin     | 0.0-1.0    | Lower = detect dimmer sources       | fixed, peaks            |
| blur_size  | 1-15 (odd) | Larger = more blur, less noise      | all                     |
| clean_size | 1-11 (odd) | Larger = more aggressive cleanup    | all                     |
| min_area   | 1-100      | Filters tiny noise                  | all except peaks        |
| max_area   | 1000-86400 | Filters large regions               | percentile, otsu, fixed |
| peak_size  | 5-31 (odd) | Size of local peak detection kernel | peaks                   |

### Typical Scenarios

- **Sparse bright sources**: percentile 96-98, margin 0.8+, min_area 5+
- **Dense dim sources**: percentile 85-90, margin 0.5-0.6, min_area 1-2
- **Noisy footage**: Increase blur_size, clean_size iterations
- **Large ROIs**: Increase max_area (otsu, fixed, percentile)

## Extension Points

### Adding New Algorithms

1. Create `core/algorithms/new_algorithm.py` with structure:

    ```python
    import cv2
    import numpy as np
    from cv2.typing import MatLike
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class NewOptions:
        """Options for new algorithm."""

        param1: int = 50
        param2: float = 0.5
        blur_size: int = 5
        clean_size: int = 3

    def new_thresholding(
        frame: MatLike, param1=50, param2=0.5, blur_size=5, clean_size=3
    ) -> MatLike:  # fmt: skip
        """Apply thresholding, return binary mask."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        # ... algorithm-specific thresholding ...
        return thresh

    def process_frame(
        frame: MatLike, options: Optional[NewOptions] = None
    ):
        """Process frame, return (thresh, contours, centers, metadata)."""

        if options is None:
            options = NewOptions()
        thresh = new_thresholding(frame, **vars(options))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # ... detection and metadata extraction ...
        return thresh, contours, centers, metadata
    ```

2. Register in `core/settings.py`:

    ```python
    from .algorithms import percentile, otsu, peaks, fixed, new_algorithm

    ALGORITHMS = {
        # ... existing algorithms ...
        "new": AlgorithmConfig(
            name="New Algorithm",
            process_frame=new_algorithm.process_frame,
            options_class=new_algorithm.NewOptions,
            params=["param1", "param2", "blur_size", "clean_size"],
        ),
    }
    ```

3. Algorithm automatically appears in Streamlit UI dropdown with dynamic parameter controls

### Custom Visualization

1. Add function to `core/utils.py`:

    ```python
    def draw_custom(
        img, data, *, colour=(255, 0, 0), thickness=1, **kwargs
    ):
        """Custom visualization function."""

        # Modify img in-place
        for item in data:
            # ... draw operations ...
    ```

2. Use in `app.py`:
    ```python
    if viz_custom and len(centers) > 0:
        draw_custom(overlay, centers, colour=(0, 255, 0))
    ```

### Streamlit Features

- Add new `st.checkbox()` for visualization toggle
- Add new `st.number_input()` or `st.slider()` for algorithm parameters
- Parameters automatically passed through `algorithm.options(**params)` dict
- All processing loop changes automatic - no manual code updates needed
