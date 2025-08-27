import numpy as np
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker.byte_tracker import TrackState
from utils.trajectory_storage import TrajectoryStorage
from typing import Optional, Dict, List, Any

class ByteTrackWrapper:
    def __init__(
        self,
        frame_rate=30,
        track_thresh=0.3,
        match_thresh=0.85,
        buffer=30,
        aspect_ratio_thresh=3.0,
        min_box_area=10,
        mot20=False,
        # Trajectory storage parameters
        enable_trajectory_storage=False,
        trajectory_output_dir="data/trajectories",
        trajectory_export_format="json",
        trajectory_export_frequency=100
    ):
        args = type('', (), {})()
        args.track_thresh = track_thresh
        args.track_buffer = buffer
        args.match_thresh = match_thresh
        args.aspect_ratio_thresh = aspect_ratio_thresh
        args.min_box_area = min_box_area
        args.mot20 = mot20
        args.frame_rate = frame_rate
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        
        # Initialize trajectory storage if enabled
        self.enable_trajectory_storage = enable_trajectory_storage
        self.trajectory_storage = None
        if enable_trajectory_storage:
            self.trajectory_storage = TrajectoryStorage(
                output_dir=trajectory_output_dir,
                export_format=trajectory_export_format,
                export_frequency=trajectory_export_frequency
            )

    def update(self, detections, frame):
        if len(detections) == 0:
            dets = np.empty((0, 6), dtype=np.float32)  # [x1, y1, x2, y2, conf, cls_id]
        else:
            dets = np.array(detections, dtype=np.float32)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1)

        online_targets = self.tracker.update(
            dets[:, :5],
            (frame.shape[0], frame.shape[1]),
            (frame.shape[0], frame.shape[1])
        )

        tracks = []
        class_mapping = {}
        
        for t in online_targets:
            tlwh = t.tlwh
            track_id = t.track_id
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h

            # Buscar la clase correspondiente a esta track
            matched_cls = None
            for det in dets:
                if np.allclose(det[:4], [x1, y1, x2, y2], atol=5.0):
                    matched_cls = int(det[5])
                    break
            
            tracks.append((int(x1), int(y1), int(x2), int(y2), track_id, matched_cls))
            
            # Store class mapping for trajectory storage
            if matched_cls is not None:
                class_mapping[track_id] = matched_cls
        
        # Update trajectory storage if enabled
        if self.enable_trajectory_storage and self.trajectory_storage:
            self.trajectory_storage.update_trajectories(online_targets, class_mapping)
        
        return tracks
    
    def get_trajectory_data(self) -> Optional[Dict[str, Any]]:
        """
        Get current trajectory data from the storage system.
        
        Returns:
            Dictionary containing trajectory data, or None if storage is disabled
        """
        if self.trajectory_storage:
            return {
                'trajectories': dict(self.trajectory_storage.trajectories),
                'metadata': self.trajectory_storage.track_metadata,
                'summary': self.trajectory_storage.get_trajectory_summary()
            }
        return None
    
    def export_trajectories(self, filename_suffix: str = ""):
        """
        Manually export trajectory data to files.
        
        Args:
            filename_suffix: Optional suffix for output filenames
        """
        if self.trajectory_storage:
            self.trajectory_storage.export_trajectories(filename_suffix)
    
    def finalize_trajectories(self):
        """
        Finalize trajectory storage and export final data.
        Call this when tracking session ends.
        """
        if self.trajectory_storage:
            self.trajectory_storage.finalize_export()
    
    def get_trajectory_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics of trajectory storage.
        
        Returns:
            Dictionary with trajectory statistics, or None if storage is disabled
        """
        if self.trajectory_storage:
            return self.trajectory_storage.get_trajectory_summary()
        return None