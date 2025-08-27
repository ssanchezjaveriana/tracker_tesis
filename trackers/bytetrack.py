import numpy as np
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker.byte_tracker import TrackState

class ByteTrackWrapper:
    def __init__(self, frame_rate=30, track_thresh=0.3, match_thresh=0.85, buffer=120):
        args = type('', (), {})()
        args.track_thresh = track_thresh # Ajustar # baja para captar personas lejanas
        args.track_buffer = buffer # Ajustar # más largo por posibles oclusiones
        args.match_thresh = match_thresh # Ajustar # más estricto para mantener IDs
        args.aspect_ratio_thresh = 3.0 # Ajustar # tolera personas acostadas o mal proyectadas
        args.min_box_area = 10 # Ajustar # descarta cosas diminutas tipo pajaritos/sombras
        args.mot20 = False
        args.frame_rate = frame_rate # Ajustar # 30 fps por los videos usados 
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