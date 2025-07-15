import json
from pathlib import Path
import cv2

# Path settings
video_path = Path("single_view.mp4")
output_path = Path("output") / video_path.stem
labels_dir = output_path / "labels"
metadata_dir = Path("metadata")
images_dir = output_path / "raw_images"

# Gather all images as frame base
images_list = sorted(images_dir.glob("*.jpg"))
num_frames = len(images_list)
img_sample = cv2.imread(str(images_list[0]))
img_h, img_w = img_sample.shape[:2]

def is_box_in_rect(box, rect):
    x1, y1, x2, y2 = box
    rx, ry, rw, rh = rect
    return (rx <= x1 <= rx+rw and rx <= x2 <= rx+rw and
            ry <= y1 <= ry+rh and ry <= y2 <= ry+rh)

def is_box_in_left_bottom_quarter(box, rect):
    rx, ry, rw, rh = rect
    lb_rx = rx
    lb_ry = ry + rh / 2
    lb_rw = rw / 2
    lb_rh = rh / 2
    lb_rect = [lb_rx, lb_ry, lb_rw, lb_rh]
    return is_box_in_rect(box, lb_rect)

def is_box_overlap(box, rect):
    x1, y1, x2, y2 = box
    rx, ry, rw, rh = rect
    x_left = max(x1, rx)
    y_top = max(y1, ry)
    x_right = min(x2, rx + rw)
    y_bottom = min(y2, ry + rh)
    return x_left < x_right and y_top < y_bottom

# Read area and rim just once
meta_area_file = metadata_dir / "frame_ball_flight_area_meta.json"
meta_rim_file = metadata_dir / "frame_rim_meta.json"
with open(meta_area_file) as f:
    area_rect = json.load(f).get("rect")
with open(meta_rim_file) as f:
    rim_rect = json.load(f).get("rect")

# Collect detection and filter info for all frames
in_area_flags = []
box_centers = []
all_best_boxes = []

for i in range(num_frames):
    img_file = images_list[i]
    frame_idx = int(img_file.stem)
    label_file = labels_dir / f"{frame_idx:05d}.txt"

    best_conf = -1
    best_box = None
    best_center = None
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
                    x1 = int((xc - w/8) * img_w)
                    y1 = int((yc - h/8) * img_h)
                    x2 = int((xc + w/8) * img_w)
                    y2 = int((yc + h/8) * img_h)
                    best_box = (x1, y1, x2, y2)
                    best_center = (int(xc * img_w), int(yc * img_h))
    # area_rect and rim_rect are global, just use
    in_area = False
    if area_rect is not None and best_box is not None:
        if is_box_in_rect(best_box, area_rect) and not is_box_in_left_bottom_quarter(best_box, area_rect):
            in_area = True
    in_area_flags.append(in_area)
    box_centers.append(best_center if in_area else None)
    all_best_boxes.append(best_box if in_area else None)

# Extract segments, allowing at most three missing detection ("gap") per segment
segments = []
i = 0
while i < len(in_area_flags):
    if in_area_flags[i]:
        seg_start = i
        j = i
        gap = 0
        while j < len(in_area_flags):
            if in_area_flags[j]:
                pass
            elif gap < 3:
                gap += 1
            else:
                break
            j += 1
        seg_end = j - 1
        # At least 20 frames
        if seg_end - seg_start + 1 >= 20:
            # At least one top-30% frame
            has_top_frame = any(
                center and center[1] < img_h * 0.3
                for center in box_centers[seg_start : seg_end + 1]
            )
            # At least one frame center in right 50% of flight_area
            has_right_half = any(
                center and center[0] >= area_rect[0] + area_rect[2] / 2
                for center in box_centers[seg_start : seg_end + 1]
            )
            if has_top_frame and has_right_half:
                segments.append((seg_start, seg_end))
        i = seg_end + 1
    else:
        i += 1

# For each segment, truncate 3 frames after rim overlap
final_segments = []
for seg_start, seg_end in segments:
    rim_touch_idx = None
    for idx in range(seg_start, seg_end + 1):
        best_box = all_best_boxes[idx]
        if best_box is not None and rim_rect is not None and is_box_overlap(best_box, rim_rect):
            rim_touch_idx = idx
            break
    # If rim touched, truncate segment after rim_touch_idx + 3
    if rim_touch_idx is not None:
        new_end = min(rim_touch_idx + 3, seg_end)
        if new_end - seg_start + 1 >= 20:
            final_segments.append((seg_start, new_end))
    else:
        final_segments.append((seg_start, seg_end))

# Output
print(f"Detected segment count: {len(final_segments)}")
for idx, (seg_start, seg_end) in enumerate(final_segments):
    start_frame = int(images_list[seg_start].stem)
    end_frame = int(images_list[seg_end].stem)
    print(f"Segment {idx+1}: Start frame = {start_frame}, End frame = {end_frame}")

segments_info = [
    {"segment_index": idx+1,
     "start_frame": int(images_list[seg_start].stem),
     "end_frame": int(images_list[seg_end].stem)}
    for idx, (seg_start, seg_end) in enumerate(final_segments)
]
with open(output_path / "ball_flight_segments.json", "w") as f:
    json.dump(segments_info, f, indent=2)
print(f"Segments info saved to {output_path / 'ball_flight_segments.json'}")
