"""
Video utilities for reading and writing video frames.
"""

import logging
from typing import List
import os

import cv2
from tqdm import tqdm


def import_video_for_analysis(video_file: str) -> List:
    """
    Reads all frames from a video file.

    Raises:
        FileNotFoundError: If video file does not exist.
        IOError: If video cannot be opened.
        ValueError: If video has no frames.
    """
    if not os.path.isfile(video_file):
        logging.error(f"Input video file not found: {video_file}")
        raise FileNotFoundError(f"Input video file not found: {video_file}")

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_file}")
        raise IOError(f"Could not open video file: {video_file}")

    frames = []
    logging.info("Importing frames from video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    if len(frames) == 0:
        logging.error(f"No frames loaded from: {video_file}")
        raise ValueError(f"No frames loaded from: {video_file}")

    logging.info(f"Imported {len(frames)} frames.")
    return frames


def write_video_full_resolution(
    video_file_in: str, video_file_out: str, frame_order: List[int], l_coresp: List[int]
):
    """
    Writes reordered video using the correct mapping of indices.

    Args:
        video_file_in (str): Input video (for resolution and fps).
        video_file_out (str): Output video path.
        frame_order (List[int]): Reordered indices over reduced (outlier-removed) frames.
        l_coresp (List[int]): Mapping from reduced frame indices to original frame indices.
    """
    if not os.path.isfile(video_file_in):
        logging.error(f"Input video not found: {video_file_in}")
        raise FileNotFoundError(f"Input video not found: {video_file_in}")

    cap = cv2.VideoCapture(video_file_in)
    if not cap.isOpened():
        logging.error(f"Cannot open input video: {video_file_in}")
        raise IOError(f"Cannot open input video: {video_file_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback if fps is zero
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()

    if frame_size[0] == 0 or frame_size[1] == 0:
        logging.error("Invalid frame dimensions detected.")
        raise ValueError("Invalid video resolution detected.")

    logging.info(f"Video Info: fps={fps}, resolution={frame_size}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(video_file_out, fourcc, fps, frame_size)

    if not out.isOpened():
        logging.error(f"Failed to open VideoWriter for {video_file_out}")
        raise IOError(f"Failed to open VideoWriter for {video_file_out}")

    logging.info("Reading original frames...")
    cap = cv2.VideoCapture(video_file_in)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if len(all_frames) == 0:
        logging.error("No frames extracted from input video.")
        raise ValueError("No frames extracted from input video.")

    logging.info("Writing reordered video...")
    try:
        for idx in tqdm(frame_order, desc="Writing Frames"):
            true_frame_idx = l_coresp[idx]
            out.write(all_frames[true_frame_idx])
    except Exception as e:
        logging.exception("Error during video writing.")
        raise RuntimeError("Video writing failed.") from e
    finally:
        out.release()

    logging.info("Video writing complete.")