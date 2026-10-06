"""
YOLOv8 tabanlı nesne tespit modülü.
Sürücü telefon kullanımı tespiti için kullanılır.
"""
import cv2
import numpy as np
from ultralytics import YOLO

class PhoneDetector:
    def __init__(self, model_path: str = "yolov8n.pt", frame_skip: int = 3):
        self.model = YOLO(model_path)
        self.class_id = 67  # COCO dataset 'cell phone'
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.last_boxes = []
        self.last_found = False
        
    def detect(self, frame: np.ndarray) -> bool:
        """Kare içinde telefon tespit eder ve varsa çizer."""
        self.frame_count += 1
        if self.frame_count % self.frame_skip == 0:
            # imgsz=320 performans için ideal
            results = self.model.predict(frame, classes=[self.class_id], imgsz=320, verbose=False)
            self.last_boxes = []
            self.last_found = False
            for r in results:
                for box in r.boxes:
                    self.last_found = True
                    self.last_boxes.append((box.xyxy[0], box.conf[0]))
        
        # Son tespit edilen kutuları çiz
        for box_xyxy, conf in self.last_boxes:
            x1, y1, x2, y2 = map(int, box_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"TELEFON KULLANIMI {float(conf):.2f}", (x1, max(10, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
        return self.last_found
