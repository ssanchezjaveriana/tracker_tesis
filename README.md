# Object Detection and Tracking System with Trajectory Analysis

This project implements a complete system for object detection, tracking, and trajectory analysis in video using YOLOv8 and ByteTrack, with clustering capabilities to identify anomalous behavior patterns.

## Table of Contents
- [Installation](#installation)
- [System Usage](#system-usage)
- [Configuration](#configuration)
- [Complete Pipeline](#complete-pipeline)
- [Trajectory Analysis](#trajectory-analysis)

## Installation

### 1. Create virtual environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install ByteTrack
```bash
git clone https://github.com/FoundationVision/ByteTrack.git byte_track_repo
cd byte_track_repo
pip install -e .
cd ..
```

### 4. Verify installation
```bash
python -c "import yolox; print('YOLOX installed successfully.')"
```

## System Usage

### Video Processing

The main script processes videos applying object detection and tracking:

```bash
python main.py --video data/input_videos/{video_name}.mp4 \
               --output data/output_videos/{video_name}.mp4 \
               --config configs/{config_file}.yaml
```

**Parameters:**
- `--video`: Input video path
- `--output`: Path where the processed video will be saved
- `--config`: YAML configuration file (see [Configuration](#configuration) section)

**Example:**
```bash
python main.py --video data/input_videos/street_view.mp4 \
               --output data/output_videos/street_view_tracked.mp4 \
               --config configs/unified.yaml
```

## Configuration

The project includes three configuration files in [configs/](configs/):

### [configs/unified.yaml](configs/unified.yaml)
Complete configuration with multiple classes, grouping of similar classes, co-movement detection, and trajectory storage.

**Main features:**
- Detection of 20 COCO classes (people, vehicles, objects)
- Grouping of similar classes (e.g., all bags → backpack)
- Class-specific parameters (confidence and minimum area)
- Trajectory storage in JSON
- Trajectory visualization with fade effect
- Person-object co-movement detection

### [configs/person_only.yaml](configs/person_only.yaml)
Simplified configuration for tracking people only.

### [configs/multiclass.yaml](configs/multiclass.yaml)
Configuration for multi-class detection without grouping.

### Key Parameters

**Detection (YOLOv8):**
- `model_path`: YOLOv8 model (e.g., "yolov8m.pt")
- `detect_classes`: List of COCO class IDs to detect
- `conf`: Confidence threshold (0.0-1.0)
- `iou`: IoU threshold for NMS

**Tracking (ByteTrack):**
- `track_thresh`: Confidence threshold to start tracks
- `match_thresh`: Threshold for detection association
- `track_buffer`: Frames a track can be inactive
- `aspect_ratio_thresh`: Aspect ratio filter
- `min_box_area`: Minimum bounding box area

**Trajectory Storage:**
```yaml
trajectory_storage:
  enable: true
  output_dir: "data/trajectories"
  export_format: "json"  # "json", "csv", or "both"
  export_frequency: 100  # Export every N frames
```

**Trajectory Visualization:**
```yaml
trajectory_visualization:
  enable: true
  tail_length: 200  # Historical points to display
  thickness: 2
  fade: true  # Fade effect
```

**Co-Movement Detection:**
```yaml
comovement_detection:
  enable: true
  proximity_threshold: 100  # Maximum distance in pixels
  min_frames: 5  # Minimum frames to confirm association
  max_gap_frames: 15  # Allowed frames without proximity
```

## Complete Pipeline

### 1. Video Processing and Trajectory Extraction

```bash
# Process video with trajectory storage enabled
python main.py --video data/input_videos/video.mp4 \
               --output data/output_videos/video_tracked.mp4 \
               --config configs/unified.yaml
```

**Outputs:**
- Processed video with visualizations: `data/output_videos/video_tracked.mp4`
- Trajectories in JSON format: `data/trajectories/*.json`

### 2. Export Trajectories to CSV

Use the notebook [export_trajectories.ipynb](export_trajectories.ipynb) to consolidate all JSON files into a single CSV:

```python
# The notebook processes in batches to optimize memory
# Generates: trayectorias_completas.csv
```

**CSV Format:**
```
track_id, frame_id, timestamp, cx, cy
1, 523, 1761105123.45, 745.5, 213.2
```

### 3. Trajectory Analysis with K-Means

Use the notebook [k_means.ipynb](k_means.ipynb) for clustering analysis:

**Process:**

1. **Feature Calculation:**
   - Position deltas (x_delta, y_delta)
   - Euclidean distance
   - Velocity and magnitude
   - Smoothing with Savitzky-Golay filter
   - Distribution histograms (10 bins per metric)

2. **Clustering:**
   - Elbow method to determine optimal K
   - K-Means with k=3
   - Validation metrics:
     - Silhouette Score
     - Davies-Bouldin Index
     - Calinski-Harabasz Index

3. **Visualization:**
   - PCA for dimensionality reduction
   - Feature heatmaps per cluster
   - Feature importance analysis

**Generated Models:**
- `modelo_kmeans.joblib`: Trained K-Means model
- `scaler_kmeans.joblib`: StandardScaler scaler
- `pca_modelo.joblib`: PCA model
- `features_combinados.csv`: Extracted features

### 4. Inference with Trained Model

Use the notebook [inference.ipynb](inference.ipynb) to classify new trajectories:

```python
# Load models
kmeans = load('modelo_kmeans.joblib')
scaler = load('scaler_kmeans.joblib')
pca = load('pca_modelo.joblib')

# Classify new trajectories
cluster_labels = kmeans.predict(scaler.transform(new_features))
```

## Trajectory Analysis

### Extracted Features

**Basic statistics (per track):**
- `total_points`: Number of points in the trajectory
- `x_delta_min/max/avg`: Horizontal displacement statistics
- `y_delta_min/max/avg`: Vertical displacement statistics
- `distance_min/max/avg`: Distance traveled statistics
- `velocity_min/max/avg`: Velocity statistics

**Histograms (40 features):**
- `x_delta_bin_0` to `x_delta_bin_9`: Horizontal movement distribution
- `y_delta_bin_0` to `y_delta_bin_9`: Vertical movement distribution
- `distance_bin_0` to `distance_bin_9`: Distance distribution
- `velocity_bin_0` to `velocity_bin_9`: Velocity distribution

### Cluster Interpretation

**Cluster 0**: Normal trajectories
- Regular and consistent movement
- Velocity and distance within expected ranges

**Cluster 1**: Anomalous/suspicious trajectories
- Represents ~0.02% of all trajectories
- Distinctive characteristics:
  - Very high total points (average ~2.4M points)
  - Unusual movement patterns
  - High separation in PCA space

**Cluster 2**: Long-duration trajectories
- Extensive trajectories but with normal movement
- Average ~623K points

### Quality Metrics

In the analysis with k=3, the following were obtained:
- **Silhouette Score**: 0.9389 (excellent separation)
- **Davies-Bouldin Index**: 0.8978 (good compactness)
- **Calinski-Harabasz Index**: 14901.70 (high cluster definition)

## Project Structure

```
.
├── configs/                 # YAML configuration files
├── data/
│   ├── input_videos/       # Input videos
│   ├── output_videos/      # Processed videos
│   └── trajectories/       # Exported trajectories (JSON)
├── detectors/              # Detection module (YOLOv8)
├── trackers/               # Tracking module (ByteTrack)
├── utils/                  # Utilities (visualization, co-movement)
├── main.py                 # Main script
├── export_trajectories.ipynb  # Export JSON → CSV
├── k_means.ipynb          # Clustering training
├── inference.ipynb        # Inference with trained model
└── requirements.txt       # Python dependencies
```

## Additional Notes

- The system supports parallel processing of multiple videos
- Trajectories are exported incrementally every N frames to optimize memory
- Class grouping improves tracking consistency
- Co-movement detection allows identification of person-object associations
- Trained models are reusable for new videos
