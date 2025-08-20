import argparse
import cv2
from detectors.yolov8_detector import YOLOv8Detector
from trackers.bytetrack import ByteTrackWrapper
from utils.visualize import draw_tracks

def run(video_path, output_path):
    detector = YOLOv8Detector("yolov8n.pt")
    tracker = ByteTrackWrapper()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        frame = draw_tracks(frame, tracks)
        out.write(frame)
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    args = parser.parse_args()

    run(args.video, args.output)