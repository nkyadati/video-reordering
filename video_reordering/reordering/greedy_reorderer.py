"""
Greedy Frame Reorderer supporting both Euclidean and Histogram distances.
"""

import logging
from typing import List

import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

from .base import BaseFrameReorderer


class GreedyReorderer(BaseFrameReorderer):
    """
    Greedy nearest-neighbor reorderer.
    """

    def __init__(self, auto_reverse: bool = True, use_histogram: bool = False, show_progress: bool = True):
        """
        Args:
            auto_reverse (bool): Automatically flip if reversed.
            use_histogram (bool): Whether to use histogram distances.
            show_progress (bool): Whether to show tqdm progress bar.
        """
        self.auto_reverse = auto_reverse
        self.use_histogram = use_histogram
        self.show_progress = show_progress
        logging.info(
            f"GreedyReorderer initialized (auto_reverse={self.auto_reverse}, use_histogram={self.use_histogram})"
        )

    def compute_distance_matrix(self, frame_list: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise distance matrix.

        Returns:
            np.ndarray: Distance matrix.
        """
        n = len(frame_list)
        Y = np.zeros((n, n))

        try:
            if self.use_histogram:
                logging.info("Computing distance matrix using histogram distances.")
                for i in range(n):
                    for j in range(n):
                        Y[i, j] = 1 - cv2.compareHist(
                            frame_list[i].astype("float32"),
                            frame_list[j].astype("float32"),
                            cv2.HISTCMP_CORREL,
                        )
            else:
                logging.info("Computing distance matrix using Euclidean distances.")
                Y = distance.cdist(frame_list, frame_list, "euclidean")

            Y /= Y.max() if Y.max() > 0 else 1

        except Exception as e:
            logging.exception("Error computing distance matrix.")
            raise RuntimeError("Distance matrix computation failed.") from e

        return Y

    def reorder(self, frame_list: np.ndarray) -> List[int]:
        """
        Computes frame ordering via greedy nearest neighbor search.
        """
        try:
            if not isinstance(frame_list, np.ndarray):
                raise TypeError("frame_list must be a numpy.ndarray")

            if len(frame_list) < 2:
                raise ValueError("Frame list must contain at least two frames to reorder.")

            logging.info("Starting greedy reordering...")

            Y = self.compute_distance_matrix(frame_list)

            start_img = 0
            for _ in range(10):
                cur_im = start_img
                seen_im = [cur_im]
                pbar = tqdm(total=len(frame_list) - 1, desc="Reordering", disable=not self.show_progress)
                while len(seen_im) < len(frame_list):
                    dists = Y[cur_im].copy()
                    dists[seen_im] = np.inf
                    cur_im = np.argmin(dists)
                    seen_im.append(cur_im)
                    pbar.update(1)
                pbar.close()

                diffs = [Y[seen_im[i]][seen_im[i + 1]] for i in range(len(seen_im) - 1)]
                start_img = seen_im[np.argmax(diffs)]

            if self.auto_reverse:
                seen_im = self.fix_reverse(seen_im, Y, frame_list)

            logging.info("Greedy reordering completed.")
            return seen_im

        except Exception as e:
            logging.exception("Greedy reordering failed.")
            raise

    def fix_reverse(self, order, Y, frame_list):
        """
        Reverse detection using:
        - Smoothness (sum of pairwise distances)
        - Normalized Predictability Variance (frame-to-frame distance variance)
        """

        def smoothness(order):
            return sum(Y[order[i], order[i + 1]] for i in range(len(order) - 1))

        def predictability_variance(frame_list, order):
            diffs = [np.linalg.norm(frame_list[order[i+1]] - frame_list[order[i]]) for i in range(len(order) - 1)]
            diffs = np.array(diffs)
            if diffs.max() > 0:
                diffs /= diffs.max()
            return np.var(diffs)

        s_fwd = smoothness(order)
        pv_fwd = predictability_variance(frame_list, order)
        s_rev = smoothness(order[::-1])
        pv_rev = predictability_variance(frame_list, order[::-1])

        logging.info(f"Reverse Check -> Forward Smoothness: {s_fwd:.4f}, Variance: {pv_fwd:.4f}")
        logging.info(f"Reverse Check -> Reverse Smoothness: {s_rev:.4f}, Variance: {pv_rev:.4f}")

        # Combined Score
        alpha, beta = 1.0, 1.0
        score_fwd = alpha * s_fwd + beta * pv_fwd
        score_rev = alpha * s_rev + beta * pv_rev

        if score_rev < score_fwd:
            logging.info("Reverse detected (smoothness + variance) — flipping.")
            return order[::-1]

        return order