"""
MAD-based Outlier Remover supporting both Euclidean and Histogram distances.
"""

import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

from .base import BaseOutlierRemover


class MADOutlierRemover(BaseOutlierRemover):
    """
    Median Absolute Deviation (MAD) based Outlier Remover.
    """

    def __init__(
        self,
        c: float = 3.0,
        outlier_save_dir: str = "outliers",
        use_histogram: bool = False,
    ):
        """
        Args:
            c (float): MAD threshold parameter.
            outlier_save_dir (str): Directory to save detected outlier frames.
            use_histogram (bool): Whether to use histogram-based distances.
        """
        self.c = c
        self.outlier_save_dir = outlier_save_dir
        self.use_histogram = use_histogram
        os.makedirs(self.outlier_save_dir, exist_ok=True)
        logging.info(
            f"MAD Outlier Remover initialized with c={self.c}, use_histogram={self.use_histogram}"
        )

    def remove_outliers(
        self, frame_list: np.ndarray, frame_list_dim: List[np.ndarray]
    ) -> Tuple[List[int], np.ndarray]:
        """
        Removes outliers based on MAD criterion.

        Returns:
            Tuple[List[int], np.ndarray]: Correspondence list and filtered features.
        """
        try:
            if not isinstance(frame_list, np.ndarray):
                logging.error("frame_list must be a numpy ndarray.")
                raise TypeError("frame_list must be a numpy ndarray.")

            if not isinstance(frame_list_dim, list):
                logging.error("frame_list_dim must be a list.")
                raise TypeError("frame_list_dim must be a list.")

            if len(frame_list) == 0:
                raise ValueError("Frame list is empty. Cannot perform outlier removal.")

            if len(frame_list) != len(frame_list_dim):
                raise ValueError("frame_list and frame_list_dim must have the same length.")

            logging.info(f"Starting outlier removal on {len(frame_list)} frames.")

            if self.use_histogram:
                logging.info("Using histogram-based MAD outlier removal.")
                med_hist = np.median(np.stack(frame_list), axis=0)
                dist_to_med = [
                    1 - cv2.compareHist(
                        med_hist.astype("float32"), h.astype("float32"), cv2.HISTCMP_CORREL
                    )
                    for h in frame_list
                ]
            else:
                logging.info("Using Euclidean-based MAD outlier removal.")
                feature_med = np.median(frame_list, axis=0)
                dist_to_med = np.linalg.norm(frame_list - feature_med, axis=1)

            med = np.median(dist_to_med)
            mad = np.median(np.abs(dist_to_med - med)) or 1e-6
            mask = np.abs(dist_to_med - med) <= self.c * mad

            excl_list = np.where(~mask)[0]
            logging.info(f"Detected {len(excl_list)} outliers.")

            for idx in excl_list:
                cv2.imwrite(
                    os.path.join(self.outlier_save_dir, f"outlier_{idx:04d}.png"),
                    frame_list_dim[idx],
                )

            l_coresp = list(range(len(frame_list)))
            for i in sorted(excl_list, reverse=True):
                frame_list = np.delete(frame_list, i, axis=0)
                del frame_list_dim[i]
                del l_coresp[i]

            if len(frame_list) < 2:
                logging.error("Too few frames remain after outlier removal.")
                raise ValueError("Too few frames remain after outlier removal.")

            logging.info(f"Remaining frames after outlier removal: {len(frame_list)}")

            return l_coresp, frame_list

        except Exception as e:
            logging.exception("MAD outlier removal failed.")
            raise