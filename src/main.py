import cv2
import json
from detection.yolo_detector import YOLODetector
from tracking.deep_sort_tracker import DeepSortTracker

def process_video(video_path, output_path, camera_name, detector, tracker):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    all_tracks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> Detection starting ")
        detections = detector.detect(frame)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> Tracking starting ")
        tracks = tracker.update(detections, frame)

        for track in tracks:
            track_id = track['id']
            x1, y1, x2, y2 = track['bbox']
            print(">>>>>>>>>>>>>>>>> Tracks == ",track)
            cv2. rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            all_tracks.append({
                "frame": frame_idx,
                "id": track_id,
                "bbox": [x1, x2, y1, y2],
                "camera": camera_name
            })
        out.write(frame)
    
    cap.release()
    out.release()
    return all_tracks

if __name__ == '__main__':
    detector = YOLODetector()
    tracker = DeepSortTracker()

    track1 = process_video(r'videos\short_video_001.mp4', r'output\tracking_camera_001.mp4', 'camera 1', detector, tracker)
    with open('output/tracks_camera1.json', 'w') as f1:
        json.dump(track1, f1, indent=2)
    
    track2 = process_video(r'videos\short_video_005.mp4', r'output\tracking_camera_005.mp4', 'camera 5', detector, tracker)
    with open('output/tracks_camera5.json', 'w') as f2:
        json.dump(track2, f2, indent=2)
    