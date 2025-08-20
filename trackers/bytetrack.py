import numpy as np
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker.byte_tracker import TrackState

class ByteTrackWrapper:
    def __init__(self, frame_rate=30, track_thresh=0.5, match_thresh=0.8, buffer=30):
        args = type('', (), {})()
        args.track_thresh = track_thresh
        args.track_buffer = buffer
        args.match_thresh = match_thresh
        args.aspect_ratio_thresh = 1.6
        args.min_box_area = 10
        args.mot20 = False
        args.frame_rate = frame_rate
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, detections, frame):
        dets = np.array(detections)
        online_targets = self.tracker.update(dets, (frame.shape[0], frame.shape[1]), (frame.shape[0], frame.shape[1]))
        tracks = []
        for t in online_targets:
            tlwh = t.tlwh
            track_id = t.track_id
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            tracks.append((int(x1), int(y1), int(x2), int(y2), track_id))
        return tracks