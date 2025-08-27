import numpy as np
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker.byte_tracker import TrackState
from typing import Dict, List, Tuple, Optional
import json

class ByteTrackWrapper:
    def __init__(self, frame_rate=30, track_thresh=0.5, match_thresh=0.8, buffer=30):
        args = type('', (), {})()
        args.track_thresh = track_thresh
        args.track_buffer = buffer
        args.match_thresh = match_thresh
        args.aspect_ratio_thresh = 1.6
        args.min_box_area = 10
        args.mot20 = False
        args.frame_rate = frame_rate
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, detections, frame):
        if len(detections) == 0:
            dets = np.empty((0, 5), dtype=np.float32)  # o (0, 6) si incluyes score+class
        else:
            dets = np.array(detections, dtype=np.float32)
            if dets.ndim == 1:  # detección única
                dets = dets.reshape(1, -1)

        online_targets = self.tracker.update(
            dets,
            (frame.shape[0], frame.shape[1]),
            (frame.shape[0], frame.shape[1])
        )

        tracks = []
        for t in online_targets:
            tlwh = t.tlwh
            track_id = t.track_id
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            tracks.append((int(x1), int(y1), int(x2), int(y2), track_id))
        return tracks
    
    def get_track_trajectories(self) -> Dict[int, List[Dict]]:
        """Get trajectories for all active tracks."""
        trajectories = {}
        
        # Get trajectories from all tracked stracks
        for track in self.tracker.tracked_stracks:
            if track.is_activated and track.get_trajectory_length() > 0:
                trajectories[track.track_id] = track.get_trajectory()
        
        # Also get trajectories from lost stracks (recent trajectories)
        for track in self.tracker.lost_stracks:
            if track.get_trajectory_length() > 0:
                trajectories[track.track_id] = track.get_trajectory()
                
        return trajectories
    
    def get_trajectory_by_id(self, track_id: int) -> Optional[List[Dict]]:
        """Get trajectory for a specific track ID."""
        # Check tracked stracks
        for track in self.tracker.tracked_stracks:
            if track.track_id == track_id:
                return track.get_trajectory()
        
        # Check lost stracks
        for track in self.tracker.lost_stracks:
            if track.track_id == track_id:
                return track.get_trajectory()
                
        return None
    
    def get_trajectory_centers_by_id(self, track_id: int) -> Optional[List[Tuple[float, float]]]:
        """Get trajectory centers for a specific track ID."""
        # Check tracked stracks
        for track in self.tracker.tracked_stracks:
            if track.track_id == track_id:
                return track.get_trajectory_centers()
        
        # Check lost stracks  
        for track in self.tracker.lost_stracks:
            if track.track_id == track_id:
                return track.get_trajectory_centers()
                
        return None
    
    def export_trajectories_json(self, filename: str) -> None:
        """Export all trajectories to a JSON file."""
        trajectories = self.get_track_trajectories()
        
        # Convert trajectories to JSON-serializable format
        export_data = {}
        for track_id, trajectory in trajectories.items():
            export_data[str(track_id)] = []
            for point in trajectory:
                # Convert TrackState enum to string and numpy arrays to lists
                json_point = {
                    'frame_id': int(point['frame_id']),
                    'timestamp': float(point['timestamp']),
                    'center': [float(point['center'][0]), float(point['center'][1])],
                    'bbox': [float(x) for x in point['bbox']],
                    'confidence': float(point['confidence']),
                    'state': int(point['state'])
                }
                export_data[str(track_id)].append(json_point)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def get_active_track_objects(self) -> List[STrack]:
        """Get direct access to STrack objects for advanced operations."""
        return [track for track in self.tracker.tracked_stracks if track.is_activated]
    
    def clear_all_trajectories(self) -> None:
        """Clear trajectory history for all tracks."""
        for track in self.tracker.tracked_stracks:
            track.clear_trajectory()
        for track in self.tracker.lost_stracks:
            track.clear_trajectory()
        for track in self.tracker.removed_stracks:
            track.clear_trajectory()