import cv2
import streamlit as st
import tempfile as tf

from pathlib import Path
from core.algorithms import ALGORITHMS
from core.draw import DRAWING

st.set_page_config(page_title="Big Brain Blob Tracker", layout="wide")


def bgr_to_hex(colour: tuple[int, int, int]) -> str:
    b, g, r = colour
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    hex_colour = hex_colour.lstrip("#")
    r = int(hex_colour[0:2], 16)
    g = int(hex_colour[2:4], 16)
    b = int(hex_colour[4:6], 16)
    return (b, g, r)


st.title("🔦 Big Brain Blob Tracker")

with st.sidebar:
    st.header("⚙️ Settings")
    st.divider()

    algo_selected = st.selectbox("Algorithm", list(ALGORITHMS.keys()), help="The blob detection algorithm") # fmt:skip
    st.subheader(f"{algo_selected.title()} Settings")

    algo = ALGORITHMS[algo_selected]
    algo_params = {}
    if "percentile" in algo.params:
        algo_params["percentile"] = st.number_input("Percentile", 0.0, 100.0, 96.0, help="Higher = fewer detections") # fmt:skip
    if "blur_size" in algo.params:
        algo_params["blur_size"] = st.slider("Blur Size", 1, 15, 5, step=2)
    if "clean_size" in algo.params:
        algo_params["clean_size"] = st.slider("Clean Size", 1, 11, 3, step=2)
    if "min_area" in algo.params:
        algo_params["min_area"] = st.number_input("Min Area", 1, 1000, 2) # fmt:skip
    if "max_area" in algo.params:
        algo_params["max_area"] = st.number_input("Max Area", 100, 1000000, 86400)
    if "margin" in algo.params:
        algo_params["margin"] = st.slider("Margin", 0.0, 1.0, 0.2, help="Intensity threshold as fraction of max") # fmt:skip

    st.divider()

    # visualisation

    stars_enabled = st.checkbox("Draw Stars", False)
    boxes_enabled = st.checkbox("Draw Boxes", False)
    lines_enabled = st.checkbox("Draw Lines", False)

    draw_params = {}
    if stars_enabled:
        st.subheader("Star Settings")
        star_colour = st.color_picker("Star Colour", "#FFFF00")
        star_size = st.slider("Star Size", 1, 20, 10)
        draw_params["stars"] = {"colour": hex_to_bgr(star_colour), "size": star_size}
    if boxes_enabled:
        st.subheader("Box Settings")
        box_colour = st.color_picker("Box Colour", "#00FF00")
        box_thickness = st.slider("Box Thickness", 1, 10, 1)
        draw_params["boxes"] = {"colour": hex_to_bgr(box_colour), "thickness": box_thickness} # fmt: skip
    if lines_enabled:
        st.subheader("Line Settings")
        line_colour = st.color_picker("Line Colour", "#FF0000")
        line_thickness = st.slider("Line Thickness", 1, 10, 1)
        draw_params["lines"] = {"colour": hex_to_bgr(line_colour), "thickness": line_thickness} # fmt: skip

    st.divider()
    st.text("made for nami (with <3)")  # attribution <3

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

    progress = st.progress(0.0, "Ready to cook!")
    if st.button("LETS COOK 🍳"):
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
            options = algo.options(**algo_params)
            metadata = algo.process_frame(frame, options) # fmt: skip

            # -- drawing --

            if stars_enabled:
                stars = DRAWING["stars"]
                stars_options = stars.options(**draw_params.get("stars", {}))
                stars.draw(overlay, metadata, stars_options, labels=metadata.get("labels")) # fmt: skip

            if boxes_enabled:
                boxes = DRAWING["boxes"]
                boxes_options = boxes.options(**draw_params.get("boxes", {}))
                boxes.draw(overlay, metadata, boxes_options, labels=metadata.get("labels")) # fmt: skip

            if lines_enabled:
                lines = DRAWING["lines"]
                lines_options = lines.options(**draw_params.get("lines", {}))
                lines.draw(overlay, metadata, lines_options)

            progress.progress((i + 1) / n_frames, f"Cooking frame {i + 1} / {n_frames}")

            out.write(overlay)

        cap.release()
        out.release()

        progress.progress(1.0, "Cooking complete!")
        st.balloons()  # celebratations

        st.divider()
        with open(output_path, "rb") as output_file:
            st.download_button(
                label="DOWNLOAD COOKED VIDEO ⬇️",
                data=output_file,
                file_name="output.avi",
                mime="video/x-msvideo",
            )
