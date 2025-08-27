#!/usr/bin/env python3
"""
Example script demonstrating trajectory storage and analysis functionality.

This script shows how to use the enhanced tracking system with trajectory storage,
analysis, and export capabilities.
"""

import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_basic_usage():
    """Example of basic trajectory storage usage."""
    print("=== Basic Trajectory Storage Example ===")
    
    from trackers.bytetrack import ByteTrackWrapper
    from utils.trajectory_utils import TrajectoryExporter, TrajectoryAnalyzer
    
    # Initialize tracker
    tracker = ByteTrackWrapper()
    
    # Simulate tracking updates (in real usage, this would be in a video processing loop)
    # tracks = tracker.update(detections, frame)
    
    # Get trajectories for analysis
    # trajectories = tracker.get_track_trajectories()
    
    # Export trajectories in different formats
    # TrajectoryExporter.export_to_json(trajectories, "trajectories.json")
    # TrajectoryExporter.export_to_csv(trajectories, "trajectories.csv")
    # TrajectoryExporter.export_to_mot_format(trajectories, "trajectories.txt")
    
    # Analyze trajectories
    # summary = TrajectoryAnalyzer.get_trajectory_summary(trajectories)
    
    print("Tracker initialized successfully!")
    print("In a real application, you would:")
    print("1. Process video frames")
    print("2. Get trajectories with tracker.get_track_trajectories()")
    print("3. Export trajectories in multiple formats")
    print("4. Analyze trajectory patterns and statistics")
    print()

def example_visualization_features():
    """Example of trajectory visualization features."""
    print("=== Trajectory Visualization Example ===")
    
    from utils.visualize import (
        draw_tracks_with_trajectories,
        draw_trajectory_paths,
        draw_trajectory_with_trail,
        draw_trajectory_heatmap,
        create_trajectory_summary_overlay
    )
    
    print("Available visualization functions:")
    print("- draw_tracks_with_trajectories: Enhanced track drawing with trails")
    print("- draw_trajectory_paths: Draw trajectory paths with fading")
    print("- draw_trajectory_with_trail: Draw trajectories with trail effect")
    print("- draw_trajectory_heatmap: Create heatmap overlay of trajectory density")
    print("- create_trajectory_summary_overlay: Show trajectory statistics overlay")
    print()

def example_analysis_capabilities():
    """Example of trajectory analysis capabilities."""
    print("=== Trajectory Analysis Example ===")
    
    from utils.trajectory_utils import TrajectoryAnalyzer
    
    # Create sample trajectory data for demonstration
    sample_trajectory = [
        {
            'frame_id': 1,
            'center': (100.0, 100.0),
            'bbox': (90.0, 90.0, 110.0, 110.0),
            'confidence': 0.9,
            'state': 1
        },
        {
            'frame_id': 2,
            'center': (110.0, 105.0),
            'bbox': (100.0, 95.0, 120.0, 115.0),
            'confidence': 0.85,
            'state': 1
        },
        {
            'frame_id': 3,
            'center': (120.0, 110.0),
            'bbox': (110.0, 100.0, 130.0, 120.0),
            'confidence': 0.8,
            'state': 1
        }
    ]
    
    # Analyze the sample trajectory
    path_length = TrajectoryAnalyzer.calculate_path_length(sample_trajectory)
    avg_speed = TrajectoryAnalyzer.calculate_average_speed(sample_trajectory)
    bounds = TrajectoryAnalyzer.get_trajectory_bounds(sample_trajectory)
    direction_changes = TrajectoryAnalyzer.calculate_direction_changes(sample_trajectory)
    
    print("Sample trajectory analysis results:")
    print(f"- Path length: {path_length:.2f} pixels")
    print(f"- Average speed: {avg_speed:.2f} pixels/frame")
    print(f"- Trajectory bounds: {bounds}")
    print(f"- Direction changes: {direction_changes}")
    print()

def example_usage_patterns():
    """Show common usage patterns for the trajectory system."""
    print("=== Common Usage Patterns ===")
    
    print("1. Basic video processing with trajectory storage:")
    print("   python main.py --video input.mp4 --output output.mp4")
    print()
    
    print("2. Process video without saving trajectory data:")
    print("   python main.py --video input.mp4 --output output.mp4 --no-trajectories")
    print()
    
    print("3. Process video without trajectory visualization:")
    print("   python main.py --video input.mp4 --output output.mp4 --no-trajectory-vis")
    print()
    
    print("4. Load and analyze existing trajectory data:")
    print("   from utils.trajectory_utils import TrajectoryExporter")
    print("   trajectories = TrajectoryExporter.load_trajectories_from_json('trajectories.json')")
    print()
    
    print("5. Custom trajectory analysis:")
    print("   from utils.trajectory_utils import TrajectoryAnalyzer")
    print("   summary = TrajectoryAnalyzer.get_trajectory_summary(trajectories)")
    print()

def example_configuration_options():
    """Show configuration options for trajectory storage."""
    print("=== Configuration Options ===")
    
    from byte_track_repo.yolox.tracker.basetrack import BaseTrack
    
    print("Trajectory storage can be configured by setting:")
    print(f"- BaseTrack.max_trajectory_length (default: {BaseTrack.max_trajectory_length})")
    print("  Controls how many trajectory points are stored per track")
    print()
    
    print("Visualization options:")
    print("- trail_length: Number of recent points to show in trail")
    print("- show_points: Whether to draw individual trajectory points")
    print("- show_speed: Whether to show speed information")
    print("- show_direction: Whether to show direction arrows")
    print("- alpha: Transparency for heatmap overlay")
    print()

if __name__ == "__main__":
    print("Trajectory Storage and Analysis System - Usage Examples")
    print("=" * 60)
    print()
    
    example_basic_usage()
    example_visualization_features()
    example_analysis_capabilities()
    example_usage_patterns()
    example_configuration_options()
    
    print("For more information:")
    print("- See tests/ directory for detailed test examples")
    print("- Check utils/trajectory_utils.py for all available functions")
    print("- Look at utils/visualize.py for visualization options")
    print("- Run main.py --help for command-line options")