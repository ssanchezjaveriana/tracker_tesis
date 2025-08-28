"""
Trajectory Storage and Export System

This module provides functionality to extract, store, and export trajectory data
from ByteTrack STrack objects without modifying the byte_track_repo.
"""

import json
import csv
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np


class TrajectoryStorage:
    """
    Manages trajectory data extraction and persistence for multi-object tracking.
    
    Extracts trajectory data from ByteTrack STrack objects and provides
    various export formats (JSON, CSV) with configurable options.
    """
    
    def __init__(self, 
                 output_dir: str = "data/trajectories",
                 export_format: str = "json",
                 export_frequency: int = 100,
                 max_memory_tracks: int = 1000):
        """
        Initialize trajectory storage system.
        
        Args:
            output_dir: Directory to save trajectory files
            export_format: Export format ('json', 'csv', 'both')
            export_frequency: Export every N frames (for memory management)
            max_memory_tracks: Maximum tracks to keep in memory
        """
        self.output_dir = output_dir
        self.export_format = export_format
        self.export_frequency = export_frequency
        self.max_memory_tracks = max_memory_tracks
        
        # Storage for trajectory data
        self.trajectories: Dict[int, List[Dict]] = defaultdict(list)
        self.track_metadata: Dict[int, Dict] = {}
        
        # Frame and export tracking
        self.current_frame = 0
        self.last_export_frame = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"TrajectoryStorage initialized: {output_dir}")
        print(f"Session ID: {self.session_id}")
    
    def extract_trajectory_from_track(self, track) -> Optional[Dict]:
        try:
            x1, y1, w, h = track.tlwh
            center_x = x1 + w / 2
            center_y = y1 + h / 2

            return {
                'track_id': track.track_id,
                'frame_id': self.current_frame,
                'timestamp': time.time(),
                'center': (center_x, center_y),
                'bbox': (x1, y1, x1 + w, y1 + h),
                'confidence': getattr(track, 'score', 0.0),
                'state': str(getattr(track, 'state', 'Tracked'))
            }
        except Exception as e:
            print(f"[ERROR] extracting trajectory: {e}")
            return None
    
    def update_trajectories(self, online_tracks: List, class_mapping: Dict[int, int] = None):
        """
        Update trajectory storage with current frame tracks.
        
        Args:
            online_tracks: List of STrack objects from ByteTracker
            class_mapping: Optional mapping of track_id to class_id
        """
        self.current_frame += 1
        
        for track in online_tracks:
            track_id = track.track_id
            
            # Extract trajectory data
            trajectory_point = self.extract_trajectory_from_track(track)
            if trajectory_point:
                # Add class information if available
                if class_mapping and track_id in class_mapping:
                    trajectory_point['class_id'] = class_mapping[track_id]
                
                # Store trajectory point
                self.trajectories[track_id].append(trajectory_point)
                
                # Update track metadata
                if track_id not in self.track_metadata:
                    self.track_metadata[track_id] = {
                        'first_frame': self.current_frame,
                        'class_id': trajectory_point.get('class_id', -1),
                        'total_points': 0
                    }
                
                self.track_metadata[track_id]['last_frame'] = self.current_frame
                self.track_metadata[track_id]['total_points'] += 1
        
        # Periodic export for memory management
        if (self.current_frame - self.last_export_frame) >= self.export_frequency:
            self.export_periodic()
    
    def export_periodic(self):
        """Export trajectories periodically to manage memory usage."""
        if len(self.trajectories) > 0:
            filename_suffix = f"_frames_{self.last_export_frame + 1}_to_{self.current_frame}"
            self.export_trajectories(filename_suffix=filename_suffix)
            
            # Clear old trajectories to free memory, keep only recent tracks
            self._cleanup_memory()
            self.last_export_frame = self.current_frame
    
    def _cleanup_memory(self):
        """Clean up memory by removing completed trajectories."""
        # Keep only tracks that have been updated in recent frames
        recent_threshold = 30  # Keep tracks from last 30 frames
        tracks_to_remove = []
        
        for track_id, trajectory in self.trajectories.items():
            if trajectory:
                last_point_frame = trajectory[-1]['frame_id']
                if self.current_frame - last_point_frame > recent_threshold:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.trajectories[track_id]
            if track_id in self.track_metadata:
                del self.track_metadata[track_id]
    
    def export_trajectories(self, filename_suffix: str = ""):
        """
        Export all stored trajectories to files.
        
        Args:
            filename_suffix: Optional suffix for output filenames
        """
        if not self.trajectories:
            return
        
        base_filename = f"trajectories_{self.session_id}{filename_suffix}"
        
        if self.export_format in ["json", "both"]:
            self._export_json(base_filename + ".json")
        
        if self.export_format in ["csv", "both"]:
            self._export_csv(base_filename + ".csv")
        
        print(f"Exported trajectories: {len(self.trajectories)} tracks, frame {self.current_frame}")
    
    def _export_json(self, filename: str):
        """Export trajectories to JSON format."""
        output_path = os.path.join(self.output_dir, filename)
        
        export_data = {
            'session_id': self.session_id,
            'export_timestamp': datetime.now().isoformat(),
            'total_frames': self.current_frame,
            'total_tracks': len(self.trajectories),
            'track_metadata': self.track_metadata,
            'trajectories': dict(self.trajectories)
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _export_csv(self, filename: str):
        """Export trajectories to CSV format."""
        output_path = os.path.join(self.output_dir, filename)
        
        # Flatten trajectory data for CSV
        rows = []
        for track_id, trajectory in self.trajectories.items():
            for point in trajectory:
                row = {
                    'track_id': track_id,
                    'frame_id': point['frame_id'],
                    'timestamp': point['timestamp'],
                    'center_x': point['center'][0] if 'center' in point else None,
                    'center_y': point['center'][1] if 'center' in point else None,
                    'bbox_x1': point['bbox'][0] if 'bbox' in point else None,
                    'bbox_y1': point['bbox'][1] if 'bbox' in point else None,
                    'bbox_x2': point['bbox'][2] if 'bbox' in point else None,
                    'bbox_y2': point['bbox'][3] if 'bbox' in point else None,
                    'confidence': point.get('confidence', 0.0),
                    'class_id': point.get('class_id', -1),
                    'state': point.get('state', 'unknown')
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of stored trajectories.
        
        Returns:
            Dictionary with trajectory statistics
        """
        total_points = sum(len(traj) for traj in self.trajectories.values())
        
        return {
            'session_id': self.session_id,
            'total_tracks': len(self.trajectories),
            'total_trajectory_points': total_points,
            'current_frame': self.current_frame,
            'active_tracks': len([t for t in self.trajectories.values() if t]),
            'export_format': self.export_format,
            'output_directory': self.output_dir
        }
    
    def finalize_export(self):
        """Final export when tracking session ends."""
        if self.trajectories:
            self.export_trajectories(filename_suffix="_final")
            print(f"Final trajectory export completed. Summary: {self.get_trajectory_summary()}")
        else:
            print("No trajectories to export.")


class TrajectoryAnalyzer:
    """Provides analysis utilities for trajectory data."""
    
    @staticmethod
    def load_trajectories_from_json(filepath: str) -> Dict:
        """Load trajectory data from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def calculate_track_statistics(trajectory_data: Dict) -> Dict:
        """Calculate basic statistics for each track."""
        stats = {}
        
        for track_id, trajectory in trajectory_data['trajectories'].items():
            if not trajectory:
                continue
                
            centers = [(point['center'][0], point['center'][1]) for point in trajectory if 'center' in point]
            
            if len(centers) < 2:
                continue
            
            # Calculate distances between consecutive points
            distances = []
            for i in range(1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[i-1][0])**2 + 
                              (centers[i][1] - centers[i-1][1])**2)
                distances.append(dist)
            
            stats[track_id] = {
                'duration_frames': len(trajectory),
                'total_distance': sum(distances),
                'average_speed': np.mean(distances) if distances else 0,
                'max_speed': max(distances) if distances else 0,
                'path_length': len(centers)
            }
        
        return stats