import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


def generate_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Generate a consistent color for a track ID."""
    np.random.seed(track_id)
    color = tuple(map(int, np.random.randint(0, 255, 3)))
    return color


def draw_tracks(frame, tracks):
    """Draw current track bounding boxes and IDs."""
    for (x1, y1, x2, y2, track_id) in tracks:
        color = generate_color_for_id(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def draw_trajectory_paths(frame, trajectories: Dict[int, List[Dict]], 
                         max_trail_length: int = 50, line_thickness: int = 2) -> np.ndarray:
    """Draw trajectory paths for all tracks.
    
    Args:
        frame: Input frame to draw on
        trajectories: Dictionary of track_id -> trajectory points
        max_trail_length: Maximum number of recent points to draw
        line_thickness: Thickness of trajectory lines
    
    Returns:
        Frame with trajectory paths drawn
    """
    for track_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue
            
        color = generate_color_for_id(track_id)
        
        # Get recent trajectory points
        recent_trajectory = trajectory[-max_trail_length:] if len(trajectory) > max_trail_length else trajectory
        
        # Draw lines connecting trajectory points
        points = [point['center'] for point in recent_trajectory]
        for i in range(1, len(points)):
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            
            # Fade line intensity based on age
            alpha = (i / len(points)) * 0.8 + 0.2  # Fade from 0.2 to 1.0
            faded_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, pt1, pt2, faded_color, line_thickness)
    
    return frame


def draw_trajectory_with_trail(frame, trajectories: Dict[int, List[Dict]], 
                              trail_length: int = 30, show_points: bool = True) -> np.ndarray:
    """Draw trajectories with a fading trail effect.
    
    Args:
        frame: Input frame to draw on
        trajectories: Dictionary of track_id -> trajectory points
        trail_length: Number of recent points to show in trail
        show_points: Whether to draw individual trajectory points
    
    Returns:
        Frame with trajectory trails drawn
    """
    for track_id, trajectory in trajectories.items():
        if len(trajectory) == 0:
            continue
            
        color = generate_color_for_id(track_id)
        
        # Get recent trajectory points
        recent_trajectory = trajectory[-trail_length:] if len(trajectory) > trail_length else trajectory
        
        # Draw trail with fading effect
        for i, point in enumerate(recent_trajectory):
            center = (int(point['center'][0]), int(point['center'][1]))
            
            # Calculate alpha based on point age (newer points are more opaque)
            alpha = (i / len(recent_trajectory)) * 0.8 + 0.2
            point_color = tuple(int(c * alpha) for c in color)
            
            # Draw trajectory point
            if show_points:
                point_radius = max(2, int(4 * alpha))
                cv2.circle(frame, center, point_radius, point_color, -1)
            
            # Draw line to next point
            if i < len(recent_trajectory) - 1:
                next_center = (int(recent_trajectory[i+1]['center'][0]), 
                             int(recent_trajectory[i+1]['center'][1]))
                cv2.line(frame, center, next_center, point_color, max(1, int(2 * alpha)))
    
    return frame


def draw_trajectory_info(frame, trajectories: Dict[int, List[Dict]], 
                        show_speed: bool = True, show_direction: bool = True) -> np.ndarray:
    """Draw trajectory information overlays.
    
    Args:
        frame: Input frame to draw on
        trajectories: Dictionary of track_id -> trajectory points
        show_speed: Whether to show speed information
        show_direction: Whether to show direction arrows
    
    Returns:
        Frame with trajectory information drawn
    """
    for track_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue
            
        color = generate_color_for_id(track_id)
        current_point = trajectory[-1]
        center = (int(current_point['center'][0]), int(current_point['center'][1]))
        
        # Calculate speed if requested
        if show_speed and len(trajectory) >= 2:
            prev_point = trajectory[-2]
            dx = current_point['center'][0] - prev_point['center'][0]
            dy = current_point['center'][1] - prev_point['center'][1]
            speed = np.sqrt(dx*dx + dy*dy)
            
            # Draw speed text
            speed_text = f"{speed:.1f}px/f"
            cv2.putText(frame, speed_text, (center[0] + 10, center[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw direction arrow if requested
        if show_direction and len(trajectory) >= 2:
            prev_point = trajectory[-2]
            dx = current_point['center'][0] - prev_point['center'][0]
            dy = current_point['center'][1] - prev_point['center'][1]
            
            if np.sqrt(dx*dx + dy*dy) > 5:  # Only draw if movement is significant
                # Normalize direction vector
                length = np.sqrt(dx*dx + dy*dy)
                dx, dy = dx/length, dy/length
                
                # Draw direction arrow
                arrow_length = 20
                end_point = (int(center[0] + dx * arrow_length), 
                           int(center[1] + dy * arrow_length))
                
                cv2.arrowedLine(frame, center, end_point, color, 2, tipLength=0.3)
    
    return frame


def draw_trajectory_heatmap(frame, trajectories: Dict[int, List[Dict]], 
                           alpha: float = 0.3) -> np.ndarray:
    """Draw a heatmap overlay showing trajectory density.
    
    Args:
        frame: Input frame to draw on
        trajectories: Dictionary of track_id -> trajectory points
        alpha: Transparency of the heatmap overlay
    
    Returns:
        Frame with heatmap overlay
    """
    if not trajectories:
        return frame
    
    # Create heatmap
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    for trajectory in trajectories.values():
        for point in trajectory:
            x, y = int(point['center'][0]), int(point['center'][1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Add gaussian blob at each point
                cv2.circle(heatmap, (x, y), 10, 1.0, -1)
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply gaussian blur
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    # Convert to color
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with original frame
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay


def draw_tracks_with_trajectories(frame, tracks: List[Tuple], 
                                 tracker_wrapper, trail_length: int = 30,
                                 show_paths: bool = True, show_info: bool = True) -> np.ndarray:
    """Enhanced track drawing with trajectory visualization.
    
    Args:
        frame: Input frame to draw on
        tracks: List of current tracks (x1, y1, x2, y2, track_id)
        tracker_wrapper: ByteTrack wrapper instance to get trajectories
        trail_length: Length of trajectory trail to show
        show_paths: Whether to show trajectory paths
        show_info: Whether to show trajectory information
    
    Returns:
        Frame with tracks and trajectories drawn
    """
    # Draw trajectory trails first (behind bounding boxes)
    if show_paths:
        trajectories = tracker_wrapper.get_track_trajectories()
        frame = draw_trajectory_with_trail(frame, trajectories, trail_length)
        
        if show_info:
            frame = draw_trajectory_info(frame, trajectories)
    
    # Draw current track bounding boxes on top
    frame = draw_tracks(frame, tracks)
    
    return frame


def create_trajectory_summary_overlay(frame, trajectory_summary: Dict[int, Dict],
                                     position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """Create an overlay showing trajectory summary statistics.
    
    Args:
        frame: Input frame to draw on
        trajectory_summary: Summary statistics from TrajectoryAnalyzer
        position: Position to start drawing text (x, y)
    
    Returns:
        Frame with summary overlay
    """
    x, y = position
    line_height = 25
    
    # Draw background rectangle
    overlay_height = len(trajectory_summary) * line_height + 40
    cv2.rectangle(frame, (x-5, y-25), (x+400, y+overlay_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (x-5, y-25), (x+400, y+overlay_height), (255, 255, 255), 1)
    
    # Title
    cv2.putText(frame, "Trajectory Summary", (x, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += line_height + 5
    
    # Summary for each track
    for track_id, stats in trajectory_summary.items():
        if stats['length_points'] > 0:
            color = generate_color_for_id(track_id)
            text = f"ID {track_id}: {stats['length_points']}pts, {stats['path_length_pixels']:.0f}px, {stats['average_speed']:.1f}px/f"
            cv2.putText(frame, text, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += line_height
    
    return frame