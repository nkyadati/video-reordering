"""
Evaluator - collection of evaluation metrics for frame reordering.
"""

import logging

import cv2
import numpy as np
from scipy.stats import kendalltau


class Evaluator:
    """
    Collection of evaluation metrics.
    """

    @staticmethod
    def smoothness(order, Y):
        """
        Computes smoothness = sum of distances between consecutive frames.
        """
        s = sum(Y[order[i], order[i + 1]] for i in range(len(order) - 1))
        logging.info(f"Smoothness = {s:.4f}")
        return s

    @staticmethod
    def pairwise_consistency(order):
        """
        Computes pairwise consistency score.
        """
        correct = sum(
            1
            for i in range(len(order))
            for j in range(i + 1, len(order))
            if order[i] < order[j]
        )
        total = len(order) * (len(order) - 1) // 2
        score = correct / total if total > 0 else 0
        logging.info(f"Pairwise Consistency = {score:.4f}")
        return score

    @staticmethod
    def optical_flow_consistency(frames):
        """
        Computes the average optical flow magnitude between consecutive frames.
        """
        logging.info("Computing optical flow consistency...")
        flow_estimator = cv2.optflow.DualTVL1OpticalFlow_create()
        total_mag = 0.0
        count = 0

        for i in range(1, len(frames)):
            gray1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flow = flow_estimator.calc(gray1, gray2, None)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
            total_mag += mag
            count += 1

        avg_flow_mag = total_mag / count if count > 0 else 0
        logging.info(f"Optical Flow Consistency = {avg_flow_mag:.4f}")
        return avg_flow_mag
