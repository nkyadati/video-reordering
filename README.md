# Video Frame Reordering Pipeline

A modular, object-oriented, and fully configurable pipeline for:
- Extracting frame-level features (CNN / Transformer / Color Histogram)
- Removing outliers via Median Absolute Deviation (MAD)
- Reordering corrupted or shuffled video frames using appearance-based Greedy Nearest-Neighbor search
- Saving the reordered video with original resolution and FPS

## ✨ Features
✔ CNN, Transformer, or Histogram-based feature extractors  
✔ Outlier detection and removal  
✔ Greedy Nearest Neighbor Frame Reordering with reverse detection  
✔ Full Logging  
✔ Parallel Feature Extraction  
✔ Timing of every stage (Feature Extraction, Outlier Removal, Reordering, Video Writing, Evaluation)

## 📂 Project Structure

```
digeiz/
├── video_reordering/
│   ├── extractor/                 # Feature extraction (CNN, Transformer, Histogram)
│   │   ├── base.py
│   │   ├── cnn_extractor.py
│   │   ├── histogram_extractor.py
│   │   └── transformer_extractor.py
│   ├── outlier_removal/           # Outlier detection and removal
│   │   ├── base.py
│   │   └── mad_remover.py
│   ├── reordering/                # Frame reordering methods
│   │   ├── base.py
│   │   └── greedy_reorderer.py
│   ├── utils/                     # Utility functions
│   │   └── video_utils.py
│   └── pipeline.py                # End-to-end pipeline
│   └── main.py                    # Main entry point
├── requirements.txt
└── [input_videos] and [output_videos] folders (see below)
```

## ✅ Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/nkyadati/video-reordering.git
cd video-reordering
```

### Step 2: Setup Conda Environment

```bash
conda create -n video_reorder_env python=3.9
conda activate video_reorder_env
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📦 Input & Output Data

Please download the input shuffled videos and store them in an appropriate directory.

➡️ **[Google Drive](https://drive.google.com/drive/folders/120z2FlDrmfunK03YYvw7-VBlW7UhjG9q?usp=sharing)**

## ▶️ Running the Pipeline

```bash
python video_reordering/main.py -vi <path_to_input_video.mp4> -vo <path_to_output_video.mp4> -m cnn
```

## 📊 Pipeline Overview

```
[Input Video]
      │
      ▼
[Frame Extraction]
      │
      ▼
[Feature Extraction]
  ├── CNN
  ├── Vision Transformer (ViT)
  └── Histogram-based
      │
      ▼
[Outlier Removal]
      │
      ▼
[Greedy Frame Reordering]
      │
      ▼
[Video Reconstruction]
      │
      ▼
[Output Video]
```

## 💡 Notes
- If using CNN or Transformer extractors, ensure CUDA is available if you want GPU acceleration.
- Histogram-based extraction is fastest but less robust for complex videos.
- All stages are modular and swappable thanks to the OOP design.
- Seed control is integrated for reproducibility.

## ✅ Example Log Output:

```
[INFO] Feature extraction   : 1.37 seconds
[INFO] Outlier removal      : 0.56 seconds
[INFO] Reordering           : 0.05 seconds
[INFO] Video writing        : 0.54 seconds
[INFO] Total                : 2.51 seconds
[INFO] ===================================
[INFO] ===== Video Reordering Pipeline Completed Successfully =====
```

## 🟣 Potential Extensions
- Having objective evaluation metrics
- Integrating optical flow for feature extraction
- Investigating hybrid features: motion + appearance