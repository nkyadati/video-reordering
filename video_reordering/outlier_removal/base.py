"""
Base class for outlier removal strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BaseOutlierRemover(ABC):
    """
    Abstract Base Class for outlier removal methods.
    """

    @abstractmethod
    def remove_outliers(
        self, frame_list: np.ndarray, frame_list_dim: List[np.ndarray]
    ) -> Tuple[List[int], np.ndarray]:
        """
        Removes outliers from the list of frame features.

        Args:
            frame_list (np.ndarray): Extracted features of frames.
            frame_list_dim (List[np.ndarray]): Original frames.

        Returns:
            Tuple[List[int], np.ndarray]:
                - l_coresp: Correspondence list after removing outliers.
                - frame_list: Frame list after removing outliers.
        """
        pass
