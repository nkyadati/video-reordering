"""
Main entrypoint for the Video Reordering Pipeline.
"""

import argparse
import logging
import sys

from extractor.cnn_extractor import CNNFeatureExtractor
from extractor.histogram_extractor import HistogramFeatureExtractor
from extractor.transformer_extractor import TransformerFeatureExtractor
from outlier_removal.mad_remover import MADOutlierRemover
from pipeline import VideoReorderingPipeline, set_seed
from reordering.greedy_reorderer import GreedyReorderer


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Video Frame Reordering Pipeline")

    parser.add_argument(
        "-vi", "--video_in", default="corrupted_video.mp4", help="Input video file path"
    )
    parser.add_argument(
        "-vo", "--video_out", default="output_video.mp4", help="Output video file path"
    )

    parser.add_argument(
        "-m",
        "--model_type",
        choices=["cnn", "transformer", "histogram"],
        default="cnn",
        help="Feature extractor type",
    )

    parser.add_argument(
        "--outlier_dir",
        default="outliers",
        help="Directory to save removed outlier frames",
    )

    parser.add_argument(
        "--results_dir", default="results", help="Directory to save evaluation report"
    )

    parser.add_argument(
        "--c", type=float, default=3.2, help="MAD threshold constant (default=3.2)"
    )

    parser.add_argument(
        "--no_auto_reverse",
        action="store_true",
        default=False,
        help="Disable automatic reverse order fixing",
    )

    return parser.parse_args()


def main():
    """
    Main function to launch the pipeline.
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    set_seed(42)

    try:
        args = parse_args()
        logging.info("=== Pipeline Configuration ===")
        logging.info(f"Input Video    : {args.video_in}")
        logging.info(f"Output Video   : {args.video_out}")
        logging.info(f"Extractor      : {args.model_type}")
        logging.info(f"Outlier Dir    : {args.outlier_dir}")
        logging.info(f"Results Dir    : {args.results_dir}")
        logging.info(f"MAD Threshold  : {args.c}")
        logging.info(f"Auto Reverse   : {not args.no_auto_reverse}")

        # ----- Select Extractor -----
        if args.model_type == "cnn":
            extractor = CNNFeatureExtractor()
            use_histogram = False
        elif args.model_type == "transformer":
            extractor = TransformerFeatureExtractor()
            use_histogram = False
        elif args.model_type == "histogram":
            extractor = HistogramFeatureExtractor()
            use_histogram = True
        else:
            logging.error(f"Unknown model type: {args.model_type}")
            sys.exit(1)

        # ----- Outlier Remover -----
        outlier_remover = MADOutlierRemover(
            c=args.c, outlier_save_dir=args.outlier_dir, use_histogram=use_histogram
        )

        # ----- Reorderer -----
        reorderer = GreedyReorderer(
            auto_reverse=not args.no_auto_reverse, use_histogram=use_histogram
        )

        # ----- Pipeline -----
        pipeline = VideoReorderingPipeline(
            extractor=extractor,
            outlier_remover=outlier_remover,
            reorderer=reorderer,
            outlier_save_dir=args.outlier_dir,
            results_dir=args.results_dir,
        )

        # ----- Process -----
        pipeline.process(args.video_in, args.video_out)
        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.exception("Pipeline execution failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()