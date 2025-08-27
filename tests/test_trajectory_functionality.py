import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from byte_track_repo.yolox.tracker.basetrack import BaseTrack, TrackState
from byte_track_repo.yolox.tracker.byte_tracker import STrack
from utils.trajectory_utils import TrajectoryExporter, TrajectoryAnalyzer
from trackers.bytetrack import ByteTrackWrapper


class TestBaseTrackTrajectory:
    """Test trajectory functionality in BaseTrack class."""
    
    def test_add_trajectory_point(self):
        """Test adding trajectory points to a track."""
        track = BaseTrack()
        track.frame_id = 1
        track.state = TrackState.Tracked
        
        bbox = (10.0, 20.0, 50.0, 80.0)
        confidence = 0.9
        
        track.add_trajectory_point(bbox, confidence, timestamp=1.0)
        
        assert track.get_trajectory_length() == 1
        trajectory = track.get_trajectory()
        assert len(trajectory) == 1
        
        point = trajectory[0]
        assert point['frame_id'] == 1
        assert point['timestamp'] == 1.0
        assert point['center'] == (30.0, 50.0)  # Center of bbox
        assert point['bbox'] == bbox
        assert point['confidence'] == confidence
        assert point['state'] == TrackState.Tracked
    
    def test_trajectory_length_limit(self):
        """Test that trajectory length is limited to prevent memory issues."""
        track = BaseTrack()
        track.max_trajectory_length = 5  # Set a small limit for testing
        track.state = TrackState.Tracked
        
        # Add more points than the limit
        for i in range(10):
            track.frame_id = i
            bbox = (float(i), float(i), float(i+10), float(i+10))
            track.add_trajectory_point(bbox, 0.8, timestamp=float(i))
        
        # Should only keep the last 5 points
        assert track.get_trajectory_length() == 5
        trajectory = track.get_trajectory()
        assert trajectory[0]['frame_id'] == 5  # Should start from frame 5
        assert trajectory[-1]['frame_id'] == 9  # Should end at frame 9
    
    def test_get_trajectory_centers(self):
        """Test getting trajectory centers."""
        track = BaseTrack()
        track.frame_id = 1
        track.state = TrackState.Tracked
        
        # Add two points
        track.add_trajectory_point((10.0, 20.0, 30.0, 40.0), 0.9, timestamp=1.0)
        track.frame_id = 2
        track.add_trajectory_point((15.0, 25.0, 35.0, 45.0), 0.8, timestamp=2.0)
        
        centers = track.get_trajectory_centers()
        assert len(centers) == 2
        assert centers[0] == (20.0, 30.0)  # Center of first bbox
        assert centers[1] == (25.0, 35.0)  # Center of second bbox
    
    def test_clear_trajectory(self):
        """Test clearing trajectory data."""
        track = BaseTrack()
        track.frame_id = 1
        track.state = TrackState.Tracked
        
        # Add a trajectory point
        track.add_trajectory_point((10.0, 20.0, 30.0, 40.0), 0.9)
        assert track.get_trajectory_length() == 1
        
        # Clear trajectory
        track.clear_trajectory()
        assert track.get_trajectory_length() == 0
        assert track.get_trajectory() == []


class TestTrajectoryExporter:
    """Test trajectory export functionality."""
    
    def create_sample_trajectories(self):
        """Create sample trajectory data for testing."""
        return {
            1: [
                {
                    'frame_id': 1,
                    'timestamp': 1.0,
                    'center': (100.0, 200.0),
                    'bbox': (90.0, 190.0, 110.0, 210.0),
                    'confidence': 0.9,
                    'state': TrackState.Tracked
                },
                {
                    'frame_id': 2,
                    'timestamp': 2.0,
                    'center': (105.0, 205.0),
                    'bbox': (95.0, 195.0, 115.0, 215.0),
                    'confidence': 0.85,
                    'state': TrackState.Tracked
                }
            ],
            2: [
                {
                    'frame_id': 1,
                    'timestamp': 1.0,
                    'center': (200.0, 300.0),
                    'bbox': (190.0, 290.0, 210.0, 310.0),
                    'confidence': 0.8,
                    'state': TrackState.Tracked
                }
            ]
        }
    
    def test_export_to_json(self):
        """Test JSON export functionality."""
        trajectories = self.create_sample_trajectories()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            TrajectoryExporter.export_to_json(trajectories, temp_path)
            
            # Verify file was created and has correct content
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert '1' in data and '2' in data
            assert len(data['1']) == 2
            assert len(data['2']) == 1
            
            # Check structure of first point
            point = data['1'][0]
            assert point['frame_id'] == 1
            assert point['center_x'] == 100.0
            assert point['center_y'] == 200.0
            assert point['confidence'] == 0.9
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_to_csv(self):
        """Test CSV export functionality."""
        trajectories = self.create_sample_trajectories()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            TrajectoryExporter.export_to_csv(trajectories, temp_path)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Read and verify content
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Should have header + 3 data rows (2 for track 1, 1 for track 2)
            assert len(lines) == 4
            assert 'track_id,frame_id,timestamp,center_x,center_y' in lines[0]
            assert '1,1,1.0,100.0,200.0' in lines[1]
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_export_to_mot_format(self):
        """Test MOT format export functionality."""
        trajectories = self.create_sample_trajectories()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            TrajectoryExporter.export_to_mot_format(trajectories, temp_path)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Read and verify content
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Should have 3 lines (2 for track 1, 1 for track 2)
            assert len(lines) == 3
            
            # Check MOT format structure (frame,id,x,y,w,h,conf,x,y,z)
            first_line = lines[0].strip().split(',')
            assert len(first_line) == 10
            assert first_line[0] == '1'  # frame_id
            assert first_line[1] == '1'  # track_id
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTrajectoryAnalyzer:
    """Test trajectory analysis functionality."""
    
    def create_linear_trajectory(self):
        """Create a simple linear trajectory for testing."""
        return [
            {
                'frame_id': 1,
                'timestamp': 1.0,
                'center': (0.0, 0.0),
                'bbox': (-10.0, -10.0, 10.0, 10.0),
                'confidence': 0.9,
                'state': TrackState.Tracked
            },
            {
                'frame_id': 2,
                'timestamp': 2.0,
                'center': (10.0, 0.0),
                'bbox': (0.0, -10.0, 20.0, 10.0),
                'confidence': 0.9,
                'state': TrackState.Tracked
            },
            {
                'frame_id': 3,
                'timestamp': 3.0,
                'center': (20.0, 0.0),
                'bbox': (10.0, -10.0, 30.0, 10.0),
                'confidence': 0.9,
                'state': TrackState.Tracked
            }
        ]
    
    def test_calculate_path_length(self):
        """Test path length calculation."""
        trajectory = self.create_linear_trajectory()
        
        # Linear trajectory with 10-pixel steps
        expected_length = 20.0  # Two 10-pixel segments
        calculated_length = TrajectoryAnalyzer.calculate_path_length(trajectory)
        
        assert abs(calculated_length - expected_length) < 0.001
    
    def test_calculate_average_speed(self):
        """Test average speed calculation."""
        trajectory = self.create_linear_trajectory()
        
        # 20 pixels over 2 frames = 10 pixels per frame
        expected_speed = 10.0
        calculated_speed = TrajectoryAnalyzer.calculate_average_speed(trajectory)
        
        assert abs(calculated_speed - expected_speed) < 0.001
    
    def test_get_trajectory_bounds(self):
        """Test trajectory bounding box calculation."""
        trajectory = self.create_linear_trajectory()
        
        bounds = TrajectoryAnalyzer.get_trajectory_bounds(trajectory)
        expected_bounds = (0.0, 0.0, 20.0, 0.0)  # x_min, y_min, x_max, y_max
        
        assert bounds == expected_bounds
    
    def test_calculate_direction_changes(self):
        """Test direction change calculation."""
        # Create a trajectory with a 90-degree turn
        trajectory = [
            {
                'frame_id': 1,
                'center': (0.0, 0.0),
                'bbox': (0, 0, 0, 0),
                'confidence': 0.9,
                'state': TrackState.Tracked
            },
            {
                'frame_id': 2,
                'center': (10.0, 0.0),
                'bbox': (0, 0, 0, 0),
                'confidence': 0.9,
                'state': TrackState.Tracked
            },
            {
                'frame_id': 3,
                'center': (10.0, 10.0),
                'bbox': (0, 0, 0, 0),
                'confidence': 0.9,
                'state': TrackState.Tracked
            }
        ]
        
        # Should detect one direction change (90 degrees)
        changes = TrajectoryAnalyzer.calculate_direction_changes(trajectory, angle_threshold=30.0)
        assert changes == 1
    
    def test_get_trajectory_summary(self):
        """Test trajectory summary generation."""
        trajectories = {
            1: self.create_linear_trajectory(),
            2: []  # Empty trajectory
        }
        
        summary = TrajectoryAnalyzer.get_trajectory_summary(trajectories)
        
        assert 1 in summary and 2 in summary
        
        # Check track 1 summary
        track1_summary = summary[1]
        assert track1_summary['length_points'] == 3
        assert track1_summary['start_frame'] == 1
        assert track1_summary['end_frame'] == 3
        assert track1_summary['duration_frames'] == 3
        assert abs(track1_summary['path_length_pixels'] - 20.0) < 0.001
        assert abs(track1_summary['average_speed'] - 10.0) < 0.001
        
        # Check track 2 summary (empty trajectory)
        track2_summary = summary[2]
        assert track2_summary['length_points'] == 0
        assert track2_summary['path_length_pixels'] == 0.0


class TestByteTrackWrapperTrajectory:
    """Test trajectory functionality in ByteTrack wrapper."""
    
    @patch('trackers.bytetrack.BYTETracker')
    def test_get_track_trajectories(self, mock_tracker_class):
        """Test getting trajectories from wrapper."""
        # Create mock tracker instance
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        # Create mock tracks with trajectories
        mock_track1 = Mock()
        mock_track1.track_id = 1
        mock_track1.is_activated = True
        mock_track1.get_trajectory_length.return_value = 2
        mock_track1.get_trajectory.return_value = [
            {'frame_id': 1, 'center': (10, 20)},
            {'frame_id': 2, 'center': (15, 25)}
        ]
        
        mock_track2 = Mock()
        mock_track2.track_id = 2
        mock_track2.is_activated = True
        mock_track2.get_trajectory_length.return_value = 0
        
        mock_tracker.tracked_stracks = [mock_track1, mock_track2]
        mock_tracker.lost_stracks = []
        
        # Create wrapper and test
        wrapper = ByteTrackWrapper()
        trajectories = wrapper.get_track_trajectories()
        
        # Should only return track 1 (has trajectory data)
        assert len(trajectories) == 1
        assert 1 in trajectories
        assert 2 not in trajectories
        assert len(trajectories[1]) == 2
    
    @patch('trackers.bytetrack.BYTETracker')
    def test_get_trajectory_by_id(self, mock_tracker_class):
        """Test getting specific trajectory by ID."""
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        mock_track = Mock()
        mock_track.track_id = 5
        mock_track.get_trajectory.return_value = [{'frame_id': 1, 'center': (10, 20)}]
        
        mock_tracker.tracked_stracks = [mock_track]
        mock_tracker.lost_stracks = []
        
        wrapper = ByteTrackWrapper()
        trajectory = wrapper.get_trajectory_by_id(5)
        
        assert trajectory is not None
        assert len(trajectory) == 1
        assert trajectory[0]['frame_id'] == 1
        
        # Test non-existent ID
        trajectory = wrapper.get_trajectory_by_id(999)
        assert trajectory is None
    
    @patch('trackers.bytetrack.BYTETracker')
    def test_export_trajectories_json(self, mock_tracker_class):
        """Test JSON export from wrapper."""
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        mock_track = Mock()
        mock_track.track_id = 1
        mock_track.is_activated = True
        mock_track.get_trajectory_length.return_value = 1
        mock_track.get_trajectory.return_value = [
            {
                'frame_id': 1,
                'timestamp': 1.0,
                'center': (10.0, 20.0),
                'bbox': (5.0, 15.0, 15.0, 25.0),
                'confidence': 0.9,
                'state': TrackState.Tracked
            }
        ]
        
        mock_tracker.tracked_stracks = [mock_track]
        mock_tracker.lost_stracks = []
        
        wrapper = ByteTrackWrapper()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            wrapper.export_trajectories_json(temp_path)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Verify content
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert '1' in data
            assert len(data['1']) == 1
            assert data['1'][0]['frame_id'] == 1
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])