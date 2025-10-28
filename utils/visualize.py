import cv2
import numpy as np

CLASS_COLORS = {
    0:  (255, 0, 0),      # person - azul
    1:  (0, 255, 255),    # bicycle - celeste
    2:  (0, 255, 0),      # car - verde
    3:  (255, 255, 0),    # motorcycle - amarillo
    5:  (255, 0, 255),    # bus - magenta
    7:  (0, 0, 128),      # truck - azul oscuro
    13: (128, 128, 0),    # bench - oliva
    24: (0, 0, 255),      # backpack - rojo
    25: (255, 128, 0),    # umbrella - naranja
    26: (128, 0, 255),    # handbag - morado
    28: (255, 255, 0),    # suitcase - amarillo
    39: (0, 128, 128),    # bottle - verde azulado
    56: (128, 0, 0),      # chair - rojo oscuro
    57: (0, 128, 0),      # couch - verde oscuro
    58: (128, 128, 255),  # potted plant - lavanda
    60: (0, 0, 128),      # dining table - azul marino
    62: (255, 0, 255),    # tv - magenta
    63: (0, 255, 128),    # laptop - verde claro
    67: (255, 200, 0),    # cell phone - dorado
    68: (150, 0, 255),    # microwave - violeta
    69: (128, 128, 128),  # oven - gris medio
    72: (100, 100, 100),  # refrigerator - gris oscuro
    73: (200, 0, 0),      # book - rojo fuerte
}

# Para clases desconocidas (por si agregas más)
DEFAULT_COLOR = (255, 255, 255)

# Para mapear los IDs de clase a nombres (opcional, para mostrar)
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    13: "bench",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    56: "chair",
    57: "couch",
    58: "potted plant",
    60: "dining table",
    62: "tv",
    63: "laptop",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    72: "refrigerator",
    73: "book"
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


def draw_associations(frame, associations, draw_connections=True,
                     connection_color=(0, 255, 0), connection_thickness=2,
                     highlight_boxes=True, show_labels=True):
    """
    Draw person-object associations on the frame.

    Args:
        frame: The video frame to draw on
        associations: List of association dictionaries with visualization data
        draw_connections: Whether to draw lines connecting associated objects
        connection_color: Color for connection lines (BGR format)
        connection_thickness: Thickness of connection lines
        highlight_boxes: Whether to highlight associated boxes
        show_labels: Whether to show association labels

    Returns:
        Modified frame with association visualizations
    """
    for assoc in associations:
        person_bbox = assoc['person_bbox']
        object_bbox = assoc['object_bbox']
        person_id = assoc['person_id']
        object_id = assoc['object_id']
        object_class = assoc['object_class']

        # Calculate center points
        p_center = (
            int((person_bbox[0] + person_bbox[2]) / 2),
            int((person_bbox[1] + person_bbox[3]) / 2)
        )
        o_center = (
            int((object_bbox[0] + object_bbox[2]) / 2),
            int((object_bbox[1] + object_bbox[3]) / 2)
        )

        # Draw connection line between centers
        if draw_connections:
            cv2.line(frame, p_center, o_center, connection_color, connection_thickness)

        # Highlight boxes with additional border
        if highlight_boxes:
            # Draw thicker outer border for person
            cv2.rectangle(
                frame,
                (person_bbox[0] - 2, person_bbox[1] - 2),
                (person_bbox[2] + 2, person_bbox[3] + 2),
                connection_color,
                3
            )

            # Draw thicker outer border for object
            cv2.rectangle(
                frame,
                (object_bbox[0] - 2, object_bbox[1] - 2),
                (object_bbox[2] + 2, object_bbox[3] + 2),
                connection_color,
                3
            )

        # Draw association label
        if show_labels:
            object_name = CLASS_NAMES.get(object_class, f"class_{object_class}")
            label = f"Person {person_id} with {object_name} {object_id}"

            # Calculate label position (above the person box)
            label_x = person_bbox[0]
            label_y = person_bbox[1] - 30

            # Ensure label is within frame bounds
            if label_y < 30:
                label_y = person_bbox[3] + 30

            # Draw label background for better visibility
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (label_x, label_y - text_height - baseline),
                (label_x + text_width, label_y + baseline),
                (0, 0, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                connection_color,
                2
            )

    return frame