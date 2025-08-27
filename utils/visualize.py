import cv2
from .coco_classes import get_class_name

def draw_tracks(frame, tracks):
    for track in tracks:
        if len(track) == 6:
            x1, y1, x2, y2, track_id, class_id = track
            class_name = get_class_name(class_id) if class_id is not None else "unknown"
            label = f"ID {track_id}: {class_name}"
        else:
            # Backward compatibility for old format
            x1, y1, x2, y2, track_id = track[:5]
            label = f"ID {track_id}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame