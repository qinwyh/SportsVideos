import json
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

DIST_THRES = 60  # Neighbor distance threshold, in pixels
RESIDUAL_THRES = 40  # Fitted-curve residual threshold, in pixels

segments_path = Path("ball_flight_segments.json")
labels_dir = Path("output/single_view/labels")
raw_images_dir = Path("output/single_view/raw_images")
rim_meta_path = Path("metadata/frame_rim_meta.json")
traj_save_dir = Path("output/2d_fitted_trajectory")
traj_save_dir.mkdir(parents=True, exist_ok=True)

with open(rim_meta_path) as f:
    rim_rect = json.load(f)["rect"]

with open(segments_path) as f:
    segments = json.load(f)

def box_iou(box1, box2):
    """Calculate IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    return inter_area / (area1 + area2 - inter_area)

def fit_parabola(xs, ys):
    """Fit a parabola y = a*x^2 + b*x + c to points (xs, ys)."""
    def parab(x, a, b, c):
        return a * x**2 + b * x + c
    params, _ = curve_fit(parab, xs, ys)
    return params

for seg in segments:
    seg_idx = seg["segment_index"]
    start_frame = seg["start_frame"]
    end_frame = seg["end_frame"]

    points = []
    boxes = []
    frame_ids = []
    last_point = None
    last_box = None

    # Step 1: Collect all ball centers for frames in the segment (with interpolation if missing)
    for frame_id in range(start_frame, end_frame + 1):
        label_file = labels_dir / f"{frame_id:05d}.txt"
        img_file = raw_images_dir / f"{frame_id:05d}.jpg"
        if not img_file.exists():
            continue
        img = cv2.imread(str(img_file))
        ih, iw = img.shape[:2]

        best_conf = -1
        center = None
        box = None

        # Get the highest confidence detection for the ball
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        _, xc, yc, w, h, conf = map(float, parts)
                        px = int(xc * iw)
                        py = int(yc * ih)
                        bw = int(w * iw)
                        bh = int(h * ih)
                        if conf > best_conf:
                            best_conf = conf
                            center = (px, py)
                            box = [px - bw // 2, py - bh // 2, bw, bh]
        # Interpolate if no detection is available
        if center is None:
            if last_point is not None:
                center = last_point
                box = last_box
            else:
                # If no previous center, search forward for a detected center
                for f2 in range(frame_id + 1, end_frame + 1):
                    label_file2 = labels_dir / f"{f2:05d}.txt"
                    if label_file2.exists():
                        with open(label_file2) as f2f:
                            for l2 in f2f:
                                p2 = l2.strip().split()
                                if len(p2) == 6:
                                    _, xc2, yc2, w2, h2, conf2 = map(float, p2)
                                    px2 = int(xc2 * iw)
                                    py2 = int(yc2 * ih)
                                    bw2 = int(w2 * iw)
                                    bh2 = int(h2 * ih)
                                    center = (px2, py2)
                                    box = [px2 - bw2 // 2, py2 - bh2 // 2, bw2, bh2]
                                    break
                    if center is not None:
                        break
        if center is not None:
            points.append(center)
            boxes.append(box)
            frame_ids.append(frame_id)
            last_point = center
            last_box = box
            # Stop collecting points if the ball box overlaps the rim
            if box is not None and rim_rect is not None:
                iou = box_iou(box, rim_rect)
                if iou > 0.01:
                    break

    if len(points) < 3:
        print(f"Segment {seg_idx:02d}: Too few points for fitting.")
        continue

    arr = np.array(points)

    # Step 2: Outlier removal by neighbor distance
    # Mark as outlier if current and previous point are too far apart
    neighbor_mask = [True]
    for i in range(1, len(arr)):
        dist = np.linalg.norm(arr[i] - arr[i-1])
        neighbor_mask.append(dist < DIST_THRES)
    neighbor_mask = np.array(neighbor_mask, dtype=bool)
    if neighbor_mask.sum() < 3:
        neighbor_mask[:] = True  # Use all points if too few remain

    # Step 3: Initial parabola fit (using neighbor-filtered points)
    xs_init = arr[neighbor_mask, 0]
    ys_init = arr[neighbor_mask, 1]
    params_init = fit_parabola(xs_init, ys_init)
    ys_fit_init = params_init[0] * xs_init**2 + params_init[1] * xs_init + params_init[2]

    # Step 4: Outlier removal by curve residual
    residuals = np.abs(ys_init - ys_fit_init)
    curve_mask = residuals < RESIDUAL_THRES
    if curve_mask.sum() < 3:
        curve_mask[:] = True

    # Step 5: Final parabola fit (using neighbor & curve inliers)
    xs_final = xs_init[curve_mask]
    ys_final = ys_init[curve_mask]
    params_final = fit_parabola(xs_final, ys_final)
    ys_fit_final = params_final[0] * xs_final**2 + params_final[1] * xs_final + params_final[2]
    fitted_points = [(int(xs_final[i]), int(ys_fit_final[i])) for i in range(len(xs_final))]

    # Step 6: Save all information as JSON
    save_path = traj_save_dir / f"segment_{seg_idx:02d}.json"
    save_dict = {
        "original_points": [list(map(int, pt)) for pt in arr],
        "neighbor_inliers": [list(map(int, pt)) for pt in arr[neighbor_mask]],
        "curve_inliers": [list(map(int, pt)) for pt in xs_final.reshape(-1, 1) if len(pt)==2], # for compatibility, can be omitted
        "parabola_params": [float(x) for x in params_final],
        "trajectory": [list(pt) for pt in fitted_points]
    }
    with open(save_path, "w") as f:
        json.dump(save_dict, f)

    # Step 7: Visualization
    plt.figure(figsize=(7, 7))
    plt.scatter(arr[:, 0], arr[:, 1], color='blue', s=30, label='All Centers')
    plt.scatter(xs_init, ys_init, color='orange', s=40, label='Neighbor Inliers')
    plt.scatter(xs_final, ys_final, color='green', s=50, label='Curve Inliers')
    plt.plot(xs_final, ys_fit_final, color='red', label='Fitted Parabola')
    plt.gca().invert_yaxis()
    plt.title(f'Segment {seg_idx:02d} Ball Trajectory')
    plt.xlabel('X (pixel)')
    plt.ylabel('Y (pixel)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(traj_save_dir / f"segment_{seg_idx:02d}_fit.png")
    plt.close()

    print(f"Segment {seg_idx:02d} done.")

print("All segments processed.")
