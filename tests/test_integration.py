import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers.bytetrack import ByteTrackWrapper
from utils.trajectory_utils import TrajectoryExporter, TrajectoryAnalyzer


class TestTrackingIntegration:
    """Integration tests for the complete tracking pipeline."""
    
    def create_mock_detections(self, num_people=2):
        """Create mock detection data for testing."""
        detections = []
        for i in range(num_people):
            # Format: [x1, y1, x2, y2, confidence]
            x1, y1 = 50 + i * 100, 50 + i * 50
            x2, y2 = x1 + 50, y1 + 80
            confidence = 0.8 + i * 0.1
            detections.append([x1, y1, x2, y2, confidence])
        return detections
    
    def create_test_frame(self, width=640, height=480):
        """Create a test frame for tracking."""
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    @patch('trackers.bytetrack.BYTETracker')
    def test_basic_tracking_pipeline(self, mock_tracker_class):
        """Test the basic tracking pipeline with trajectory storage."""
        # Create mock BYTETracker
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        # Create mock STrack objects with trajectory functionality
        mock_track1 = Mock()
        mock_track1.track_id = 1
        mock_track1.is_activated = True
        mock_track1.tlwh = [10, 20, 50, 80]  # x, y, w, h
        mock_track1.get_trajectory_length.return_value = 3
        mock_track1.get_trajectory.return_value = [
            {'frame_id': 1, 'center': (35, 60), 'confidence': 0.9},
            {'frame_id': 2, 'center': (40, 65), 'confidence': 0.85},
            {'frame_id': 3, 'center': (45, 70), 'confidence': 0.8}
        ]
        mock_track1.get_trajectory_centers.return_value = [(35, 60), (40, 65), (45, 70)]
        
        mock_track2 = Mock()
        mock_track2.track_id = 2
        mock_track2.is_activated = True
        mock_track2.tlwh = [150, 120, 50, 80]
        mock_track2.get_trajectory_length.return_value = 2
        mock_track2.get_trajectory.return_value = [
            {'frame_id': 2, 'center': (175, 160), 'confidence': 0.95},
            {'frame_id': 3, 'center': (170, 155), 'confidence': 0.9}
        ]
        mock_track2.get_trajectory_centers.return_value = [(175, 160), (170, 155)]
        
        mock_tracker.update.return_value = [mock_track1, mock_track2]
        mock_tracker.tracked_stracks = [mock_track1, mock_track2]
        mock_tracker.lost_stracks = []
        
        # Test the wrapper
        wrapper = ByteTrackWrapper()
        frame = self.create_test_frame()
        detections = self.create_mock_detections()
        
        # Update tracker
        tracks = wrapper.update(detections, frame)
        
        # Verify basic tracking output
        assert len(tracks) == 2
        track_ids = [track[4] for track in tracks]
        assert 1 in track_ids and 2 in track_ids
        
        # Test trajectory functionality
        trajectories = wrapper.get_track_trajectories()
        assert len(trajectories) == 2
        assert 1 in trajectories and 2 in trajectories
        
        # Verify trajectory data
        track1_trajectory = trajectories[1]
        assert len(track1_trajectory) == 3
        assert track1_trajectory[0]['frame_id'] == 1
        
        track2_trajectory = trajectories[2]
        assert len(track2_trajectory) == 2
        assert track2_trajectory[0]['frame_id'] == 2
    
    @patch('trackers.bytetrack.BYTETracker')
    def test_trajectory_export_integration(self, mock_tracker_class):
        """Test trajectory export in an integrated workflow."""
        # Setup mocks
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        # Create realistic trajectory data
        trajectory_data = {
            1: [
                {
                    'frame_id': 1,
                    'timestamp': 1.0,
                    'center': (100.0, 200.0),
                    'bbox': (75.0, 175.0, 125.0, 225.0),
                    'confidence': 0.9,
                    'state': 1  # TrackState.Tracked
                },
                {
                    'frame_id': 2,
                    'timestamp': 2.0,
                    'center': (105.0, 205.0),
                    'bbox': (80.0, 180.0, 130.0, 230.0),
                    'confidence': 0.85,
                    'state': 1
                }
            ],
            2: [
                {
                    'frame_id': 1,
                    'timestamp': 1.0,
                    'center': (300.0, 400.0),
                    'bbox': (275.0, 375.0, 325.0, 425.0),
                    'confidence': 0.8,
                    'state': 1
                }
            ]
        }
        
        # Test all export formats
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "test_trajectories.json")
            csv_path = os.path.join(temp_dir, "test_trajectories.csv")
            mot_path = os.path.join(temp_dir, "test_trajectories.txt")
            
            # Export in all formats
            TrajectoryExporter.export_to_json(trajectory_data, json_path)
            TrajectoryExporter.export_to_csv(trajectory_data, csv_path)
            TrajectoryExporter.export_to_mot_format(trajectory_data, mot_path)
            
            # Verify all files were created
            assert os.path.exists(json_path)
            assert os.path.exists(csv_path)
            assert os.path.exists(mot_path)
            
            # Test loading back from JSON
            loaded_trajectories = TrajectoryExporter.load_trajectories_from_json(json_path)
            assert len(loaded_trajectories) == 2
            assert 1 in loaded_trajectories and 2 in loaded_trajectories
            assert len(loaded_trajectories[1]) == 2
            assert len(loaded_trajectories[2]) == 1
    
    def test_trajectory_analysis_integration(self):
        """Test trajectory analysis with realistic data."""
        # Create a curved trajectory (person walking in an arc)
        trajectory_data = {
            1: [
                {'frame_id': 1, 'center': (0.0, 0.0), 'confidence': 0.9, 'state': 1},
                {'frame_id': 2, 'center': (10.0, 5.0), 'confidence': 0.9, 'state': 1},
                {'frame_id': 3, 'center': (18.0, 15.0), 'confidence': 0.85, 'state': 1},
                {'frame_id': 4, 'center': (20.0, 25.0), 'confidence': 0.8, 'state': 1},
                {'frame_id': 5, 'center': (15.0, 35.0), 'confidence': 0.8, 'state': 1}
            ],
            2: [
                {'frame_id': 1, 'center': (100.0, 100.0), 'confidence': 0.95, 'state': 1},
                {'frame_id': 2, 'center': (110.0, 100.0), 'confidence': 0.9, 'state': 1}
            ]
        }
        
        # Analyze trajectories
        summary = TrajectoryAnalyzer.get_trajectory_summary(trajectory_data)
        
        # Verify analysis results
        assert len(summary) == 2
        
        # Track 1 should have more complex movement
        track1_summary = summary[1]
        assert track1_summary['length_points'] == 5
        assert track1_summary['duration_frames'] == 5
        assert track1_summary['path_length_pixels'] > 0
        assert track1_summary['average_speed'] > 0
        assert track1_summary['direction_changes'] >= 0
        
        # Track 2 should have simple linear movement
        track2_summary = summary[2]
        assert track2_summary['length_points'] == 2
        assert track2_summary['duration_frames'] == 2
        assert abs(track2_summary['path_length_pixels'] - 10.0) < 0.01  # 10 pixels horizontal
        assert abs(track2_summary['average_speed'] - 10.0) < 0.01  # 10 pixels per frame
    
    def test_empty_trajectory_handling(self):
        """Test handling of tracks with no trajectory data."""
        wrapper = ByteTrackWrapper()
        
        # Test with empty trajectories
        trajectories = wrapper.get_track_trajectories()
        assert trajectories == {}
        
        # Test trajectory analysis with empty data
        summary = TrajectoryAnalyzer.get_trajectory_summary({})
        assert summary == {}
        
        # Test trajectory export with empty data
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "empty_trajectories.json")
            TrajectoryExporter.export_to_json({}, json_path)
            
            # File should exist but be empty
            assert os.path.exists(json_path)
            with open(json_path, 'r') as f:
                import json
                data = json.load(f)
                assert data == {}
    
    @pytest.mark.slow
    def test_memory_usage_with_long_trajectories(self):
        """Test memory usage with very long trajectories."""
        wrapper = ByteTrackWrapper()
        
        # This test would require actual tracking over many frames
        # For now, we just test that the trajectory length limit works
        from byte_track_repo.yolox.tracker.basetrack import BaseTrack
        
        track = BaseTrack()
        track.max_trajectory_length = 100
        
        # Add many trajectory points
        for i in range(200):
            track.frame_id = i
            track.add_trajectory_point(
                (float(i), float(i), float(i+10), float(i+10)), 
                0.8, 
                timestamp=float(i)
            )
        
        # Should be limited to max_trajectory_length
        assert track.get_trajectory_length() <= track.max_trajectory_length
        assert track.get_trajectory_length() == 100


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])