from ultralytics import YOLO
import torch

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt", detect_classes={0}, conf=0.25, iou=0.45,
                 class_conf_thresholds=None, class_min_box_areas=None, class_grouping=None):
        print(f"[INFO] Cargando modelo en {'GPU' if torch.cuda.is_available() else 'CPU'}")
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.detect_classes = set(detect_classes)
        self.iou = iou  # NMS IoU threshold

        # Class grouping/remapping (e.g., merge backpack, handbag, suitcase into one class)
        self.class_grouping = class_grouping if class_grouping else {}
        if self.class_grouping:
            print(f"[INFO] Agrupamiento de clases habilitado:")
            for original, target in self.class_grouping.items():
                if original != target:
                    print(f"  - Clase {original} -> {target}")

        # Per-class confidence thresholds
        self.class_conf_thresholds = class_conf_thresholds if class_conf_thresholds else {}
        self.default_conf = conf

        # Per-class minimum box areas
        self.class_min_box_areas = class_min_box_areas if class_min_box_areas else {}
        self.default_min_box_area = 0

        # Use minimum confidence threshold for YOLO prediction to capture all potential detections
        # We'll filter by per-class thresholds after prediction
        if self.class_conf_thresholds:
            self.yolo_conf = min(min(self.class_conf_thresholds.values()), self.default_conf)
        else:
            self.yolo_conf = self.default_conf

        print(f"[INFO] Usando conf mínimo de {self.yolo_conf} para detección YOLO")
        if self.class_conf_thresholds:
            print(f"[INFO] Umbrales de confianza por clase: {self.class_conf_thresholds}")

    def detect(self, frame):
        results = self.model.predict(frame, device=self.device, verbose=False, conf=self.yolo_conf, iou=self.iou)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls.cpu().numpy())
            if cls in self.detect_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy())

                # Apply class grouping/remapping BEFORE other filtering
                # This ensures handbag (26) and suitcase (28) are treated as backpack (24)
                original_cls = cls
                cls = self.class_grouping.get(cls, cls)

                # Get class-specific confidence threshold (use remapped class)
                class_conf_threshold = self.class_conf_thresholds.get(cls, self.default_conf)

                # Apply per-class confidence filtering
                if conf < class_conf_threshold:
                    continue

                # Get class-specific minimum box area (use remapped class)
                class_min_box_area = self.class_min_box_areas.get(cls, self.default_min_box_area)

                # Apply per-class box area filtering
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < class_min_box_area:
                    continue

                # Use remapped class in detection output
                detections.append([x1, y1, x2, y2, conf, cls])
        return detections