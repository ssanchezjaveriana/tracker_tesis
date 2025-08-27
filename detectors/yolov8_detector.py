from ultralytics import YOLO
import torch

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt"):
        print(f"[INFO] Cargando modelo en {'GPU' if torch.cuda.is_available() else 'CPU'}")
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Classes to detect: tv, laptop, car, suitcase, refrigerator, backpack
        self.target_classes = {2, 24, 28, 62, 63, 72}  # car, backpack, suitcase, tv, laptop, refrigerator

    def detect(self, frame):
        results = self.model(frame, device=self.device, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls.cpu().numpy())
            if cls in self.target_classes:  # Detect selected big objects
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy())
                detections.append([x1, y1, x2, y2, conf])
        return detections