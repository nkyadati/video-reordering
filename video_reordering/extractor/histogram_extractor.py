"""
Histogram-based Feature Extractor.
"""

import logging

import cv2
import numpy as np

from .base import BaseExtractor


class HistogramFeatureExtractor(BaseExtractor):
    """
    Histogram-based Feature Extractor using color histograms.
    """

    def __init__(self, bins=(8, 8, 8)):
        logging.info("Using Color Histogram for feature extraction.")
        self.bins = bins

    def extract_feature(self, frame):
        """Extracts color histogram feature vector from a frame."""
        try:
            if frame is None:
                logging.error("Received None frame for histogram extraction.")
                raise ValueError("Frame is None.")

            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logging.error(f"Invalid frame shape: {frame.shape}")
                raise ValueError(f"Expected 3-channel color image, got shape {frame.shape}")

            hist = cv2.calcHist([frame], [0, 1, 2], None, self.bins,
                                [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            feature = hist.flatten()

            if np.isnan(feature).any():
                logging.error("NaNs detected in extracted histogram feature.")
                raise ValueError("Extracted feature contains NaNs.")

            return feature
        except Exception as e:
            logging.exception("Histogram feature extraction failed.")
            raise RuntimeError("Histogram feature extraction failed.") from e
