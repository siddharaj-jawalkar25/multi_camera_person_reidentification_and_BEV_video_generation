from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.4)
    
    def update(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        output_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            x1, y1, x2, y2 = int(l), int(t), int(l + w), int(t + h)
            output_tracks.append({
                "id": track_id,
                "bbox": [x1, y1, x2, y2]
            })
        return output_tracks