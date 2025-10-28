# Configuration Guide

This document explains the configuration system for the object tracking pipeline.

## Configuration Formats

### Unified Configuration (Recommended)

The new unified configuration format allows you to specify different parameters for different object classes. This is particularly useful when you need different confidence thresholds for persons vs. objects.

**File:** `configs/unified.yaml`

```yaml
model_path: "yolov8m.pt"
detect_classes: [0, 1, 2, 3, 5, 7, 13, 24, 25, 26, 28, 56, 57, 58, 60, 62, 63, 68, 69, 72]

# Per-class detection parameters
class_parameters:
  0:  # persons (class ID 0)
    conf: 0.45  # Higher confidence threshold for persons
    min_box_area: 10  # Smaller minimum box area for persons
  default:  # all other classes
    conf: 0.20  # Lower confidence threshold for objects
    min_box_area: 30  # Larger minimum box area for objects

# Global YOLO detection parameters
iou: 0.4

# ByteTrack parameters
track_thresh: 0.3
match_thresh: 0.9
track_buffer: 180
aspect_ratio_thresh: 3.0
min_box_area: 30
```

#### How Per-Class Parameters Work

1. **Class-Specific Thresholds:** You can define specific `conf` and `min_box_area` for individual class IDs
2. **Default Values:** The `default` key specifies parameters for all classes not explicitly listed
3. **Automatic Application:** During detection, each class uses its specific parameters
   - Class 0 (persons) will only be detected if confidence > 0.45
   - Classes > 0 (objects) will only be detected if confidence > 0.20

### Legacy Configuration Format

The old configuration format is still supported for backward compatibility.

**Example:** `configs/person_only.yaml`, `configs/multiclass.yaml`

```yaml
model_path: "yolov8m.pt"
detect_classes: [0]
conf: 0.45  # Single confidence threshold for all classes
iou: 0.4

track_thresh: 0.3
match_thresh: 0.95
track_buffer: 180
aspect_ratio_thresh: 3.0
min_box_area: 10
```

In this format:
- A single `conf` value applies to ALL classes in `detect_classes`
- No per-class customization is available

## Configuration Parameters

### Detection Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | string | Path to YOLOv8 model file (e.g., "yolov8m.pt") |
| `detect_classes` | list[int] | COCO class IDs to detect (0=person, 1=bicycle, 2=car, etc.) |
| `conf` | float | Default confidence threshold (0.0-1.0) |
| `iou` | float | Non-maximum suppression IoU threshold (0.0-1.0) |

### Per-Class Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `class_parameters.<class_id>.conf` | float | Confidence threshold for specific class |
| `class_parameters.<class_id>.min_box_area` | int | Minimum bounding box area in pixels |
| `class_parameters.default.conf` | float | Default confidence for unlisted classes |
| `class_parameters.default.min_box_area` | int | Default minimum box area for unlisted classes |

### ByteTrack Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `track_thresh` | float | Detection confidence threshold for tracking |
| `match_thresh` | float | IoU threshold for matching detections to tracks |
| `track_buffer` | int | Number of frames to keep tracks alive without detections |
| `aspect_ratio_thresh` | float | Maximum aspect ratio for bounding boxes |
| `min_box_area` | int | Global minimum box area (used by ByteTrack, overridden during detection by class_parameters) |

### Trajectory Storage

| Parameter | Type | Description |
|-----------|------|-------------|
| `trajectory_storage.enable` | bool | Enable trajectory data storage |
| `trajectory_storage.output_dir` | string | Directory for trajectory exports |
| `trajectory_storage.export_format` | string | "json", "csv", or "both" |
| `trajectory_storage.export_frequency` | int | Export every N frames |
| `trajectory_storage.max_memory_tracks` | int | Maximum tracks to keep in memory |

### Trajectory Visualization

| Parameter | Type | Description |
|-----------|------|-------------|
| `trajectory_visualization.enable` | bool | Draw trajectories on video |
| `trajectory_visualization.tail_length` | int | Number of historical points to display |
| `trajectory_visualization.thickness` | int | Line thickness for trajectories |
| `trajectory_visualization.fade` | bool | Apply fading effect to older points |

## Usage

To use a configuration file:

```bash
python main.py --video input.mp4 --output output.mp4 --config configs/unified.yaml
```

## COCO Class IDs Reference

Common class IDs used in this project:

- 0: person
- 1: bicycle
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck
- 13: bench
- 24: backpack
- 25: umbrella
- 26: handbag
- 28: suitcase
- 56: chair
- 57: couch
- 58: potted plant
- 60: dining table
- 62: tv
- 63: laptop
- 68: microwave
- 69: oven
- 72: refrigerator

## Migration Guide

### From Two Configs to One Unified Config

**Before:** You had separate configs for persons and objects

- `person_only.yaml`: conf=0.45, only class 0
- `multiclass.yaml`: conf=0.20, multiple classes

**After:** Use one unified config

```yaml
detect_classes: [0, 1, 2, ...]  # All classes you want

class_parameters:
  0:
    conf: 0.45  # Stricter for persons
    min_box_area: 10
  default:
    conf: 0.20  # More lenient for objects
    min_box_area: 30
```

**Benefits:**
- Single configuration file to maintain
- Different detection parameters per class
- Cleaner workflow
- Better organization

## Implementation Details

### How Confidence Filtering Works

1. **YOLO Detection Phase:**
   - Uses the minimum confidence threshold across all classes
   - This ensures no potential detections are missed early

2. **Post-Processing Phase:**
   - Each detection is filtered by its class-specific confidence threshold
   - Detections below their class threshold are discarded

3. **Box Area Filtering:**
   - Each detection is also filtered by its class-specific minimum box area
   - Helps remove false positives from small or noisy detections

This two-phase approach ensures optimal detection while maintaining per-class control.
