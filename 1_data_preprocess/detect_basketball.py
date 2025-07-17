import os
import cv2
import subprocess
import numpy as np

from pathlib import Path
from tqdm import tqdm


# All file directories
video_path = Path("input/single_view.mp4")
image_path = Path("input") / video_path.stem
raw_images_path = image_path / "raw_images"
raw_images_path.mkdir(parents=True, exist_ok=True)

# Extract all frames from video_path and save as jpg images to raw_images_path
print("Extracting images from the video:")
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# total_frames = 660

frame_idx = 1
with tqdm(total=total_frames, desc="Extracting frames") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > total_frames:
            break
        img_name = f"{frame_idx:05d}.jpg"
        img_path = raw_images_path / img_name
        cv2.imwrite(str(img_path), frame)
        frame_idx += 1
        pbar.update(1)
cap.release()
print(f"Extracted {frame_idx-1} frames to {raw_images_path}")


# Detect basketball by YOLO_v5
detect_cmd = [
        "python",
        "yolov5/detect.py",
        "--weights",
        "weight/basketball_detection.pt",
        "--source",
        f"{raw_images_path}",
        "--save-txt",
        "--save-conf",
        # "--nosave",
        "--project",
        f"{image_path.parent}",
        "--name",
        f"{video_path.stem}",
        "--exist-ok",
        "--device",
        "cuda:0",
    ]
subprocess.run(detect_cmd, check=True)
