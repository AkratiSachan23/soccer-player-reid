from ultralytics import YOLO
import numpy as np

class PlayerDetector:
    def __init__(self, model_path, conf_thresh=0.5, player_class_id=2):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.player_class_id = player_class_id
    
    def detect(self, frame):
        results = self.model(frame)[0]
        boxes = []
        for x1, y1, x2, y2, conf, cls in results.boxes.data.cpu().numpy():
            if int(cls) == self.player_class_id and conf > self.conf_thresh:
                boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return boxes