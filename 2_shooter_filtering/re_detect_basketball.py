import os
import json
import subprocess
from pathlib import Path
import shutil

# Path configurations
segment_json = Path("output/single_view/ball_flight_segments.json")
masked_dir = Path("output/single_view/masked_segments")
labels_dir = Path("output/single_view/labels")
yolov5_path = Path("yolov5/detect.py")
weights_path = Path("weight/basketball_detection.pt")

# Read all segment definitions (start/end frame, segment_index, etc.)
with open(segment_json, "r") as f:
    segments = json.load(f)

for seg in segments:
    seg_idx = seg['segment_index']
    start_frame = seg['start_frame']
    end_frame = seg['end_frame']
    num_frames = end_frame - start_frame + 1

    # Masked video for this segment
    masked_video = masked_dir / f"masked_single_view_{seg_idx:03d}.mp4"
    # Temporary YOLO output directory for this segment
    yolo_out_dir = masked_dir / f"segment_{seg_idx:03d}_yolo"
    yolo_labels_dir = yolo_out_dir / "labels"
    
    # Run YOLO detection on the masked video segment
    detect_cmd = [
        "python", str(yolov5_path),
        "--weights", str(weights_path),
        "--source", str(masked_video),
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--project", str(yolo_out_dir.parent),
        "--name", yolo_out_dir.name,
        "--exist-ok",
        "--device", "cuda:0"
    ]
    subprocess.run(detect_cmd, check=True)

    # Rename and move each label file according to its global frame number
    label_files = sorted(yolo_labels_dir.glob("*.txt"))
    for i, label_file in enumerate(label_files):
        global_frame_idx = start_frame + i
        out_label_path = labels_dir / f"{global_frame_idx:05d}.txt"
        shutil.copy(label_file, out_label_path)

    # Optionally remove the temporary YOLO output directory for this segment
    shutil.rmtree(yolo_out_dir)

print("All segments processed, labels saved to output/single_view/labels/ with global frame numbers.")
