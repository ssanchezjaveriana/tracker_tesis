import cv2

def draw_tracks(frame, tracks):
    for (x1, y1, x2, y2, track_id) in tracks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame