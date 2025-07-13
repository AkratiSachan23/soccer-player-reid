import cv2
import time
import yaml
from pathlib import Path
from .detector import PlayerDetector
from .feature_extractor import FeatureExtractor
from .tracker import Tracker

class PlayerReIDPipeline:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        
        self.detector = PlayerDetector(
            self.cfg['detection']['model_path'],
            self.cfg['detection']['conf_thresh'],
            self.cfg['detection']['player_class_id']
        )
        self.feature_extractor = FeatureExtractor()
        self.tracker = Tracker(
            self.cfg['tracking']['max_age'],
            self.cfg['tracking']['match_threshold']
        )
    
    def process_video(self):
        cap = cv2.VideoCapture(self.cfg['io']['input_video'])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_path = Path(self.cfg['io']['output_video'])
        output_path.parent.mkdir(exist_ok=True)
        
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            t0 = time.time()
            detections = self._process_frame(frame)
            self._visualize(frame, detections)
            writer.write(frame)
            print(f"Frame processed in {(time.time()-t0)*1000:.1f}ms")
        
        cap.release()
        writer.release()
    
    def _process_frame(self, frame):
        boxes = self.detector.detect(frame)
        detections = []
        for (x1, y1, x2, y2, conf) in boxes:
            crop = frame[y1:y2, x1:x2]
            feat = self.feature_extractor.extract(crop)
            detections.append({'tlbr': (x1, y1, x2, y2), 'feat': feat})
        return self.tracker.update(detections)
    
    def _visualize(self, frame, detections):
        for state, tid in detections:
            x1, y1, x2, y2 = map(int, state[:4])
            
            # More professional looking box
            box_color = (0, 200, 0)  # Softer green
            text_color = (255, 255, 255)  # White text
            
            # Draw box with transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)  # Filled
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)  # Apply transparency
            
            # Draw border
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
            
            # Draw ID with background for readability
            text = f"ID:{tid}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1-text_height-5), (x1+text_width, y1), box_color, -1)
            cv2.putText(frame, text, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)