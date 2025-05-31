from ultralytics import YOLO
import torch
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolov12n.pt', conf_thres=0.5):
        self.model = YOLO(model_path)
        self.model.conf = conf_thres

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detection = []
        for result in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = result[:6]
            if int(cls) == 0:
                detection.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], float(conf), 'person'))
        return detection