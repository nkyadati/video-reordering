"""
CNN-based Feature Extractor using ResNet18.
"""

import logging

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

from .base import BaseExtractor


class CNNFeatureExtractor(BaseExtractor):
    """
    CNN-based Feature Extractor using ResNet18.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(
            f"Using ResNet18 (CNN) for feature extraction on {self.device.upper()}."
        )
        self.model = resnet18(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_feature(self, frame):
        """Extracts CNN feature vector from a frame."""
        try:
            if frame is None:
                logging.error("Received None frame for feature extraction.")
                raise ValueError("Frame is None.")

            tensor = self.transform(frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model(tensor).squeeze().cpu().numpy()
            if np.isnan(feature).any():
                logging.error("NaNs detected in extracted CNN feature.")
                raise ValueError("Extracted feature contains NaNs.")
            return feature
        except Exception as e:
            logging.exception("CNN feature extraction failed.")
            raise RuntimeError("CNN feature extraction failed.") from e
