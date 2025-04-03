"""
Base class for frame reordering strategies.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseFrameReorderer(ABC):
    """
    Abstract Base Class for frame reorderers.
    """

    @abstractmethod
    def reorder(self, frame_list: np.ndarray) -> List[int]:
        """
        Computes an optimal frame order given the extracted features.

        Args:
            frame_list (np.ndarray): Extracted feature matrix.

        Returns:
            List[int]: The computed ordering of frame indices.
        """
        pass
