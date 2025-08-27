import json
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path


class TrajectoryExporter:
    """Utility class for exporting trajectory data in various formats."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def export_to_json(trajectories: Dict[int, List[Dict]], filename: str) -> None:
        """Export trajectories to JSON format."""
        export_data = {}
        
        for track_id, trajectory in trajectories.items():
            export_data[str(track_id)] = []
            for point in trajectory:
                json_point = {
                    'frame_id': int(point['frame_id']),
                    'timestamp': float(point['timestamp']),
                    'center_x': float(point['center'][0]),
                    'center_y': float(point['center'][1]),
                    'bbox_x1': float(point['bbox'][0]),
                    'bbox_y1': float(point['bbox'][1]),
                    'bbox_x2': float(point['bbox'][2]),
                    'bbox_y2': float(point['bbox'][3]),
                    'confidence': float(point['confidence']),
                    'state': int(point['state'])
                }
                export_data[str(track_id)].append(json_point)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    @staticmethod
    def export_to_csv(trajectories: Dict[int, List[Dict]], filename: str) -> None:
        """Export trajectories to CSV format."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            fieldnames = ['track_id', 'frame_id', 'timestamp', 'center_x', 'center_y', 
                         'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence', 'state']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for track_id, trajectory in trajectories.items():
                for point in trajectory:
                    row = {
                        'track_id': track_id,
                        'frame_id': int(point['frame_id']),
                        'timestamp': float(point['timestamp']),
                        'center_x': float(point['center'][0]),
                        'center_y': float(point['center'][1]),
                        'bbox_x1': float(point['bbox'][0]),
                        'bbox_y1': float(point['bbox'][1]),
                        'bbox_x2': float(point['bbox'][2]),
                        'bbox_y2': float(point['bbox'][3]),
                        'confidence': float(point['confidence']),
                        'state': int(point['state'])
                    }
                    writer.writerow(row)
    
    @staticmethod
    def export_to_mot_format(trajectories: Dict[int, List[Dict]], filename: str) -> None:
        """Export trajectories to MOT challenge format.
        
        MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            for track_id, trajectory in trajectories.items():
                for point in trajectory:
                    x1, y1, x2, y2 = point['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    
                    # MOT format line: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
                    line = f"{point['frame_id']},{track_id},{x1},{y1},{width},{height},{point['confidence']},-1,-1,-1\n"
                    f.write(line)
    
    @staticmethod
    def load_trajectories_from_json(filename: str) -> Dict[int, List[Dict]]:
        """Load trajectories from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        trajectories = {}
        for track_id_str, trajectory in data.items():
            track_id = int(track_id_str)
            reconstructed_trajectory = []
            
            for point in trajectory:
                reconstructed_point = {
                    'frame_id': point['frame_id'],
                    'timestamp': point['timestamp'],
                    'center': (point['center_x'], point['center_y']),
                    'bbox': (point['bbox_x1'], point['bbox_y1'], point['bbox_x2'], point['bbox_y2']),
                    'confidence': point['confidence'],
                    'state': point['state']
                }
                reconstructed_trajectory.append(reconstructed_point)
            
            trajectories[track_id] = reconstructed_trajectory
        
        return trajectories


class TrajectoryAnalyzer:
    """Utility class for analyzing trajectory data."""
    
    @staticmethod
    def calculate_path_length(trajectory: List[Dict]) -> float:
        """Calculate the total path length of a trajectory."""
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            x1, y1 = trajectory[i-1]['center']
            x2, y2 = trajectory[i]['center']
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        return total_distance
    
    @staticmethod
    def calculate_average_speed(trajectory: List[Dict]) -> float:
        """Calculate the average speed of a trajectory (pixels per frame)."""
        if len(trajectory) < 2:
            return 0.0
        
        path_length = TrajectoryAnalyzer.calculate_path_length(trajectory)
        time_span = trajectory[-1]['frame_id'] - trajectory[0]['frame_id']
        
        if time_span == 0:
            return 0.0
        
        return path_length / time_span
    
    @staticmethod
    def get_trajectory_bounds(trajectory: List[Dict]) -> Tuple[float, float, float, float]:
        """Get the bounding box that encompasses the entire trajectory."""
        if not trajectory:
            return 0.0, 0.0, 0.0, 0.0
        
        centers = [point['center'] for point in trajectory]
        x_coords = [center[0] for center in centers]
        y_coords = [center[1] for center in centers]
        
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    
    @staticmethod
    def calculate_direction_changes(trajectory: List[Dict], angle_threshold: float = 30.0) -> int:
        """Calculate the number of significant direction changes in a trajectory."""
        if len(trajectory) < 3:
            return 0
        
        direction_changes = 0
        
        for i in range(2, len(trajectory)):
            # Calculate vectors
            x1, y1 = trajectory[i-2]['center']
            x2, y2 = trajectory[i-1]['center']
            x3, y3 = trajectory[i]['center']
            
            # Vector from point i-2 to i-1
            v1 = np.array([x2 - x1, y2 - y1])
            # Vector from point i-1 to i
            v2 = np.array([x3 - x2, y3 - y2])
            
            # Skip if either vector is too short
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
            
            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms > 0:
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                
                if angle > angle_threshold:
                    direction_changes += 1
        
        return direction_changes
    
    @staticmethod
    def get_trajectory_summary(trajectories: Dict[int, List[Dict]]) -> Dict[int, Dict[str, Any]]:
        """Get summary statistics for all trajectories."""
        summary = {}
        
        for track_id, trajectory in trajectories.items():
            if trajectory:
                summary[track_id] = {
                    'length_points': len(trajectory),
                    'path_length_pixels': TrajectoryAnalyzer.calculate_path_length(trajectory),
                    'average_speed': TrajectoryAnalyzer.calculate_average_speed(trajectory),
                    'start_frame': trajectory[0]['frame_id'],
                    'end_frame': trajectory[-1]['frame_id'],
                    'duration_frames': trajectory[-1]['frame_id'] - trajectory[0]['frame_id'] + 1,
                    'bounds': TrajectoryAnalyzer.get_trajectory_bounds(trajectory),
                    'direction_changes': TrajectoryAnalyzer.calculate_direction_changes(trajectory),
                    'average_confidence': np.mean([point['confidence'] for point in trajectory])
                }
            else:
                summary[track_id] = {
                    'length_points': 0,
                    'path_length_pixels': 0.0,
                    'average_speed': 0.0,
                    'start_frame': 0,
                    'end_frame': 0,
                    'duration_frames': 0,
                    'bounds': (0, 0, 0, 0),
                    'direction_changes': 0,
                    'average_confidence': 0.0
                }
        
        return summary