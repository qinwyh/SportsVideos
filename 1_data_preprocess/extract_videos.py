import cv2
import json
from pathlib import Path

# Path settings
video_path = Path("input/single_view.mp4")
segments_json_path = Path("output/single_view/ball_flight_segments.json")
output_dir = Path("output/single_view/segments")
output_dir.mkdir(parents=True, exist_ok=True)

# Load segments info
with open(segments_json_path, "r") as f:
    segments = json.load(f)

# Open the video and get its parameters
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Build a list of all frames (assuming all segment frame indices are valid)
# (OpenCV uses 0-based frame indexing)
for seg in segments:
    seg_idx = seg["segment_index"]
    start_frame = seg["start_frame"]
    end_frame = seg["end_frame"]

    out_path = output_dir / f"single_view_{seg_idx:03d}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_id in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: failed to read frame {frame_id} for segment {seg_idx}")
            break
        out.write(frame)
    out.release()
    print(f"Saved: {out_path}")

cap.release()
print("All segments extracted.")
