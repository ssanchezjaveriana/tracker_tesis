import cv2
import numpy as np

# Colores RGB para cada clase
CLASS_COLORS = {
    0: (255, 0, 0),      # person - azul
    2: (0, 255, 0),      # car - verde
    24: (0, 0, 255),     # backpack - rojo
    28: (255, 255, 0),   # suitcase - cyan
    62: (255, 0, 255),   # tv - magenta
    63: (0, 255, 255),   # laptop - amarillo
    72: (100, 100, 100)  # refrigerator - gris
}

# Para clases desconocidas (por si agregas más)
DEFAULT_COLOR = (255, 255, 255)

# Para mapear los IDs de clase a nombres (opcional, para mostrar)
CLASS_NAMES = {
    0: "person",
    2: "car",
    24: "backpack",
    28: "suitcase",
    62: "tv",
    63: "laptop",
    72: "refrigerator"
}

def draw_trajectory(frame, trajectory_points, color, tail_length=30, thickness=2, fade=True):
    """
    Draw trajectory path for a single track with optional fading effect.
    
    Args:
        frame: The video frame to draw on
        trajectory_points: List of (x, y) center points
        color: RGB color tuple for the trajectory
        tail_length: Maximum number of points to display
        thickness: Line thickness
        fade: Whether to apply fading effect to older points
    """
    if len(trajectory_points) < 2:
        return frame
    
    # Limit trajectory points to tail_length
    points_to_draw = trajectory_points[-tail_length:] if len(trajectory_points) > tail_length else trajectory_points
    
    # Convert to numpy array for OpenCV
    pts = np.array([(int(p[0]), int(p[1])) for p in points_to_draw], np.int32)
    
    if fade:
        # Draw with fading effect - older points are more transparent
        for i in range(1, len(pts)):
            # Calculate opacity based on position in trajectory
            alpha = float(i) / len(pts)
            # Adjust thickness based on position (thinner for older points)
            current_thickness = max(1, int(thickness * alpha))
            # Draw line segment
            cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), color, current_thickness)
    else:
        # Draw continuous trajectory without fading
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, color, thickness)
    
    return frame

def draw_tracks(frame, tracks, tracker=None, draw_trajectories=True, 
                trajectory_tail_length=30, trajectory_thickness=2, trajectory_fade=True):
    """
    Draw tracks and optionally their trajectories on the frame.
    
    Args:
        frame: The video frame to draw on
        tracks: List of tracks (x1, y1, x2, y2, track_id, cls_id)
        tracker: ByteTrackWrapper instance for accessing trajectory data
        draw_trajectories: Whether to draw trajectories
        trajectory_tail_length: Number of historical points to display
        trajectory_thickness: Line thickness for trajectories
        trajectory_fade: Whether to use fading effect
    """
    # Draw trajectories first (so they appear behind boxes)
    if draw_trajectories and tracker is not None:
        try:
            # Get trajectory data from tracker
            if hasattr(tracker, 'trajectory_storage') and tracker.trajectory_storage is not None:
                trajectories = tracker.trajectory_storage.trajectories
                
                # Draw trajectory for each active track
                for track in tracks:
                    _, _, _, _, track_id, cls_id = track
                    
                    if track_id in trajectories:
                        # Get trajectory points
                        trajectory_data = trajectories[track_id]
                        # Extract center points
                        centers = [point['center'] for point in trajectory_data]
                        
                        if len(centers) > 1:
                            # Get color for this class
                            color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
                            # Draw trajectory
                            frame = draw_trajectory(
                                frame, centers, color,
                                tail_length=trajectory_tail_length,
                                thickness=trajectory_thickness,
                                fade=trajectory_fade
                            )
        except Exception as e:
            # Silently handle errors to not break visualization
            pass
    
    # Draw bounding boxes and labels (on top of trajectories)
    for track in tracks:
        x1, y1, x2, y2, track_id, cls_id = track
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        label = f"{CLASS_NAMES.get(cls_id, 'class')} ID:{track_id}"
        
        # Caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Etiqueta
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame