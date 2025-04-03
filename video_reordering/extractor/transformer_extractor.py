"""
Transformer-based Feature Extractor using ViT-B/16.
"""

import logging

import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as transforms

from .base import BaseExtractor


class TransformerFeatureExtractor(BaseExtractor):
    """
    Transformer-based Feature Extractor using ViT-B/16.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(
            f"Using ViT-B/16 (Transformer) for feature extraction on {self.device.upper()}."
        )
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True).to(
            self.device
        )
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def extract_feature(self, frame):
        """Extracts Transformer feature vector from a frame."""
        try:
            if frame is None:
                logging.error("Received None frame for feature extraction.")
                raise ValueError("Frame is None.")

            tensor = self.transform(frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model(tensor).squeeze().cpu().numpy()

            if np.isnan(feature).any():
                logging.error("NaNs detected in extracted Transformer feature.")
                raise ValueError("Extracted feature contains NaNs.")

            return feature
        except Exception as e:
            logging.exception("Transformer feature extraction failed.")
            raise RuntimeError("Transformer feature extraction failed.") from e
