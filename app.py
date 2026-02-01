import cv2
import streamlit as st
import tempfile as tf

from pathlib import Path
from core.settings import ALGORITHMS
from core.utils import draw_stars, draw_boxes, draw_lines

st.set_page_config(page_title="Big Brain Blob Tracker", layout="wide")

st.title("🔦 Big Brain Blob Tracker")

with st.sidebar:
    st.header("⚙️ Settings")

    selection = st.selectbox("Algorithm", list(ALGORITHMS.keys()), help="The blob detection algorithm") # fmt:skip

    st.divider()

    st.subheader(f"{selection.title()} Settings")

    algorithm = ALGORITHMS[selection]

    params = {}
    if "percentile" in algorithm.params:
        params["percentile"] = st.number_input("Percentile", 0.0, 100.0, 96.0, help="Higher = fewer detections") # fmt:skip
    if "blur_size" in algorithm.params:
        params["blur_size"] = st.slider("Blur Size", 1, 15, 5, step=2)
    if "clean_size" in algorithm.params:
        params["clean_size"] = st.slider("Clean Size", 1, 11, 3, step=2)
    if "min_area" in algorithm.params:
        params["min_area"] = st.number_input("Min Area", 1, 1000, 2) # fmt:skip
    if "max_area" in algorithm.params:
        params["max_area"] = st.number_input("Max Area", 100, 1000000, 86400)
    if "margin" in algorithm.params:
        params["margin"] = st.slider("Margin", 0.0, 1.0, 0.2, help="Intensity threshold as fraction of max") # fmt:skip

    st.divider()

    # visualisation

    st.subheader("Visualization")

    viz_stars = st.checkbox("Draw Stars", value=True)
    viz_boxes = st.checkbox("Draw Boxes", value=True)
    viz_lines = st.checkbox("Draw Lines", value=False)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location

    suffix = Path(uploaded_file.name).suffix

    with tf.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        input_path = temp_file.name

    with tf.NamedTemporaryFile(delete=False, suffix=".avi") as output_file:
        output_path = output_file.name

    st.success("File uploaded successfully!")

    if st.button("LETS COOK! 🍳"):
        progress = st.progress(0.0)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Error: Could not open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0  # default fps

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter.fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                st.error("Error: Could not read frame.")
                break

            # -- processing --

            overlay = frame.copy()
            options = algorithm.options(**params)
            thresh, contours, centers, metadata = algorithm.process_frame(frame, options) # fmt: skip

            # -- drawing --

            if viz_stars and len(centers) > 0:
                draw_stars(overlay, centers, metadata["boxes"], labels=metadata.get("labels")) # fmt: skip

            if viz_boxes and len(metadata["boxes"]) > 0:
                draw_boxes(overlay, metadata["boxes"], labels=metadata.get("labels"))

            if viz_lines and len(centers) > 1:
                draw_lines(overlay, centers)

            progress.progress((i + 1) / n_frames)

            out.write(overlay)

        cap.release()
        out.release()

        progress.progress(1.0)

        with open(output_path, "rb") as output_file:
            st.download_button(
                label="Download Processed Video",
                data=output_file,
                file_name="output.avi",
                mime="video/x-msvideo",
            )
