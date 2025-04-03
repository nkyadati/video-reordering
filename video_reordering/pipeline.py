"""
VideoReorderingPipeline - high-level orchestrator for the video reordering process.
"""

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from extractor.base import BaseExtractor
from outlier_removal.base import BaseOutlierRemover
from reordering.base import BaseFrameReorderer
from utils import video_utils


def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class VideoReorderingPipeline:
    """The complete pipeline to process and reorder video frames."""

    def __init__(
        self,
        extractor: BaseExtractor,
        outlier_remover: BaseOutlierRemover,
        reorderer: BaseFrameReorderer,
        outlier_save_dir: str = "outliers",
        results_dir: str = "results",
    ):
        self.extractor = extractor
        self.outlier_remover = outlier_remover
        self.reorderer = reorderer
        self.outlier_save_dir = outlier_save_dir
        self.results_dir = results_dir

        os.makedirs(self.outlier_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def process(self, video_in: str, video_out: str):
        """Executes the full pipeline and measures inference times."""

        set_seed(42)
        try:
            logging.info("===== Video Reordering Pipeline Started =====")

            if not os.path.isfile(video_in):
                logging.error(f"Input video not found: {video_in}")
                raise FileNotFoundError(f"Input video not found: {video_in}")

            # --------------------------
            # Import Video
            # --------------------------
            frames = video_utils.import_video_for_analysis(video_in)
            timings = {}

            # --------------------------
            # Feature Extraction
            # --------------------------
            logging.info("Extracting features from frames...")
            start_time = time.perf_counter()
            try:
                with ThreadPoolExecutor() as executor:
                    features = list(executor.map(self.extractor.extract_feature, frames))
            except Exception as e:
                logging.exception("Feature extraction failed.")
                raise
            feature_extraction_time = time.perf_counter() - start_time
            timings["feature extraction"] = feature_extraction_time

            frame_list = np.stack(features).astype(np.float32)

            # --------------------------
            # Outlier Removal
            # --------------------------
            start_time = time.perf_counter()
            try:
                l_coresp, frame_list = self.outlier_remover.remove_outliers(frame_list, frames)
            except Exception as e:
                logging.exception("Outlier removal failed.")
                raise
            outlier_removal_time = time.perf_counter() - start_time
            timings["outlier removal"] = outlier_removal_time

            if frame_list.shape[0] < 2:
                logging.error("Not enough frames after outlier removal to continue.")
                raise ValueError("Pipeline stopped: insufficient frames after outlier removal.")

            # --------------------------
            # Frame Reordering
            # --------------------------
            start_time = time.perf_counter()
            try:
                frame_order = self.reorderer.reorder(frame_list)
            except Exception as e:
                logging.exception("Frame reordering failed.")
                raise
            reordering_time = time.perf_counter() - start_time
            timings["reordering"] = reordering_time

            # --------------------------
            # Video Writing
            # --------------------------
            start_time = time.perf_counter()
            try:
                video_utils.write_video_full_resolution(video_in, video_out, frame_order, l_coresp)
            except Exception as e:
                logging.exception("Video writing failed.")
                raise
            video_writing_time = time.perf_counter() - start_time
            timings["video writing"] = video_writing_time
            
            # --------------------------
            # Timing Summary
            # --------------------------

            total_time = sum(timings.values())
            logging.info("\n===== Pipeline Timing Summary =====")
            for step, duration in timings.items():
                logging.info(f"{step.capitalize():<20} : {duration:.2f} seconds")
            logging.info(f"{'Total':<20} : {total_time:.2f} seconds")
            logging.info("===================================")

            logging.info("===== Video Reordering Pipeline Completed Successfully =====")

        except Exception as e:
            logging.exception("Pipeline failed with an unexpected error.")
            raise