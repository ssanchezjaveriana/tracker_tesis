from ultralytics import YOLO
import torch

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt", detect_classes={0}, conf=0.25, iou=0.45):
        print(f"[INFO] Cargando modelo en {'GPU' if torch.cuda.is_available() else 'CPU'}")
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.detect_classes = set(detect_classes)
        self.conf = conf  # confidence threshold
        self.iou = iou  # NMS IoU threshold

    def detect(self, frame):
        results = self.model.predict(frame, device=self.device, verbose=False, conf=self.conf, iou=self.iou)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls.cpu().numpy())
            if cls in self.detect_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy())
                detections.append([x1, y1, x2, y2, conf, cls])
        return detections