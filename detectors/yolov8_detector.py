from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls.cpu().numpy())
            if cls == 0:  # Solo personas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy())
                detections.append([x1, y1, x2, y2, conf])
        return detections