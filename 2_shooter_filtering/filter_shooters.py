import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Load start_frame for first five segments to calculate average release point
segments_json = Path("ball_flight_segments.json")
labels_dir = Path("output/single_view/labels")
raw_images_dir = Path("output/single_view/raw_images")
with open(segments_json) as f:
    segments = json.load(f)
start_frames = [seg["start_frame"] for seg in segments[:5]]

# Compute average ball position for those frames
img_sample = cv2.imread(str(next(raw_images_dir.glob("*.jpg"))))
img_h, img_w = img_sample.shape[:2]
ball_positions = []
for frame_idx in start_frames:
    label_file = labels_dir / f"{frame_idx:05d}.txt"
    best_conf = -1
    best_xy = None
    if label_file.exists():
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    xc, yc, w, h, conf = map(float, parts[1:6])
                elif len(parts) >= 5:
                    xc, yc, w, h = map(float, parts[1:5])
                    conf = 0
                else:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    x = int(xc * img_w)
                    y = int(yc * img_h)
                    best_xy = (x, y)
    if best_xy:
        ball_positions.append(best_xy)

if not ball_positions:
    raise RuntimeError("No ball positions found in the first five segments!")

avg_release = tuple(np.mean(ball_positions, axis=0).astype(int))
print(f"Average release point: {avg_release}")

input_videos_dir = Path("single_segments")
output_videos_dir = Path("masked_single_segments")
output_videos_dir.mkdir(parents=True, exist_ok=True)

yolo = YOLO('yolov8x.pt')  # Use default confidence threshold

PADDING = 10  # Shooter box padding in pixels


def boxes_overlap(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    return not (x12 <= x21 or x11 >= x22 or y12 <= y21 or y11 >= y22)

# This variable records the last mask positions to continue masking when no non-shooter detected
last_non_shooter_boxes = None

video_files = sorted(input_videos_dir.glob("*.mp4"))
for video_file in video_files:
    print(f"Processing {video_file.name}")
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    masked_name = 'masked_' + video_file.name
    out_path = output_videos_dir / masked_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo(frame)
        person_boxes = []
        for det in results[0].boxes:
            if int(det.cls) == 0:  # use YOLO's default conf threshold
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                person_boxes.append(((x1, y1, x2, y2), (cx, cy)))
        show = frame.copy()
        # for box, _ in person_boxes:
            # cv2.rectangle(show, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        if len(person_boxes) > 1:
            # Mask new non-shooter regions
            dists = [np.linalg.norm(np.array(center) - np.array(avg_release)) for _, center in person_boxes]
            shooter_idx = int(np.argmin(dists))
            x1, y1, x2, y2 = person_boxes[shooter_idx][0]
            sx1 = max(0, x1 - PADDING)
            sy1 = max(0, y1 - PADDING)
            sx2 = min(width, x2 + PADDING)
            sy2 = min(height, y2 + PADDING)
            shooter_box = (sx1, sy1, sx2, sy2)
            # cv2.rectangle(show, (sx1, sy1), (sx2, sy2), (0,0,255), 2)
            mask = show.copy()
            non_shooter_boxes = []
            for idx, (box, center) in enumerate(person_boxes):
                if idx == shooter_idx:
                    continue
                non_shooter_boxes.append(box)
                if not boxes_overlap(box, shooter_box):
                    x1, y1, x2, y2 = box
                    mask[y1:y2, x1:x2] = 128
                else:
                    x1, y1, x2, y2 = box
                    bx1c = max(x1, 0)
                    by1c = max(y1, 0)
                    bx2c = min(x2, width)
                    by2c = min(y2, height)
                    # Top
                    if by1c < sy1:
                        mask[by1c:sy1, bx1c:bx2c] = 128
                    # Bottom
                    if by2c > sy2:
                        mask[sy2:by2c, bx1c:bx2c] = 128
                    # Left
                    if bx1c < sx1:
                        mask[max(by1c, sy1):min(by2c, sy2), bx1c:sx1] = 128
                    # Right
                    if bx2c > sx2:
                        mask[max(by1c, sy1):min(by2c, sy2), sx2:bx2c] = 128
            last_non_shooter_boxes = non_shooter_boxes.copy()
            out.write(mask)
        elif len(person_boxes) == 1:
            # No non-shooter, repeat last mask positions if available
            x1, y1, x2, y2 = person_boxes[0][0]
            sx1 = max(0, x1 - PADDING)
            sy1 = max(0, y1 - PADDING)
            sx2 = min(width, x2 + PADDING)
            sy2 = min(height, y2 + PADDING)
            # cv2.rectangle(show, (sx1, sy1), (sx2, sy2), (0,0,255), 2)
            mask = show.copy()
            if last_non_shooter_boxes:
                for box in last_non_shooter_boxes:
                    x1, y1, x2, y2 = box
                    mask[y1:y2, x1:x2] = 128
            out.write(mask)
        else:
            # If nobody detected, output original frame
            out.write(frame)
    cap.release()
    out.release()
    print(f"Masked video saved to {out_path}")
print("All videos processed.")
