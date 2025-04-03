"""
Base class for feature extractors.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseExtractor(ABC):
    """
    Abstract Base Class for feature extractors.

    All concrete extractors must implement the `extract_feature` method.
    """

    @abstractmethod
    def extract_feature(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract feature from a frame.

        Raises:
            ValueError: If the input frame is empty or None.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Received empty frame for feature extraction.")

        pass
