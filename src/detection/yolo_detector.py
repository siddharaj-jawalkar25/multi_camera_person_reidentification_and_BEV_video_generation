from ultralytics import YOLO
import torch
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolo11s.pt', conf_thres=0.65):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
    
    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy()

        height, width = frame.shape[:2]
        detections = []
        for box, conf, cls in zip(boxes, confs, clses):
            if int(cls) == 0 and conf >= self.conf_thres:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    detections.append(([x1, y1, w, h], float(conf), 'person'))
        return detections