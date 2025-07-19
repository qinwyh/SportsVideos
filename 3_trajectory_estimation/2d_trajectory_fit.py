import json
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

DIST_THRES = 100  # Neighbor distance threshold, in pixels
RESIDUAL_THRES = 40  # Fitted-curve residual threshold, in pixels

raw_images_dir = Path("input/single_view/raw_images")
segments_path = Path("output/single_view/ball_flight_segments.json")
labels_dir = Path("output/single_view/labels")
rim_meta_path = Path("metadata/frame_rim_meta.json")
traj_save_dir = Path("output/single_view/2d_fitted_trajectory")
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

def remove_outliers(points, threshold=3.0):
    """Remove outlier points based on distance jumps (using MAD)."""
    arr = np.array(points)
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff))
    keep = np.ones(len(arr), dtype=bool)
    for i, d in enumerate(diffs):
        if mad > 0 and np.abs(d - median_diff) / mad > threshold:
            keep[i+1] = False  # Mark the large-jump point as outlier
    filtered = arr[keep]
    return filtered

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
        if center is not None and center[0] > int(iw * 0.1):
            # invert y axis
            points.append(center)
            boxes.append(box)
            frame_ids.append(frame_id)
            last_point = center
            last_box = box
            if box is not None and rim_rect is not None:
                iou = box_iou(box, rim_rect)
                if iou > 0.01:
                    break

    if len(points) < 3:
        print(f"Segment {seg_idx:02d}: Too few points for fitting.")
        continue

    arr = np.array(points)

    # Step 2: Only perform outlier detection/removal on the first and last points
    def remove_edge_outlier(arr, degree=2, std_thres=2.5):
        """
        Remove outlier for start/end points if they are far from the global fitted curve.
        Keeps at least 3 points to ensure fitting is possible.
        """
        if len(arr) <= 3:
            return arr
        x = arr[:, 0]
        y = arr[:, 1]
        # Fit parabola to all points
        params = np.polyfit(x, y, degree)
        y_fit = np.polyval(params, x)
        residuals = np.abs(y - y_fit)
        mean = np.mean(residuals)
        std = np.std(residuals)
        edge_mask = np.ones(len(arr), dtype=bool)
        # Remove first point if its residual is much larger than mean
        if residuals[0] > mean + std_thres * std:
            edge_mask[0] = False
        # Remove last point if its residual is much larger than mean
        if residuals[-1] > mean + std_thres * std:
            edge_mask[-1] = False
        # Ensure at least 3 points remain
        if edge_mask.sum() < 3:
            edge_mask[:] = True
        return arr[edge_mask]

    arr_final = remove_edge_outlier(arr, degree=2, std_thres=2.5)

    # Step 3: Parabolic fit with the final filtered points
    x = arr_final[:, 0]
    y = arr_final[:, 1]
    params = np.polyfit(x, y, 2)
    fitted_func = np.poly1d(params)

     # Step 4: Visualization and save
    x_fit = np.linspace(arr_final[:, 0].min(), arr_final[:, 0].max(), 100)
    y_fit = fitted_func(x_fit)

    plt.figure(figsize=(7, 7))
    plt.scatter(arr[:, 0], arr[:, 1], color='gray', s=30, label='Raw Points')
    plt.scatter(arr_final[:, 0], arr_final[:, 1], color='blue', s=50, label='Final Inliers')
    plt.plot(x_fit, y_fit, color='red', label='Fitted Parabola')
    plt.gca().invert_yaxis()
    plt.title(f'Segment {seg_idx:02d} Ball Trajectory')
    plt.xlabel('X (pixel)')
    plt.ylabel('Y (pixel)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(traj_save_dir / f"segment_{seg_idx:02d}_fit.png")
    plt.close()

    # Save JSON with all information
    N_traj = 100  # Number of fitted points to save
    trajectory = np.stack([x_fit, y_fit], axis=1).tolist()
    save_dict = {
        "original_points": [list(map(int, pt)) for pt in arr],
        "trajectory": trajectory,
        "parabola_params": [float(p) for p in params]
    }
    save_path = traj_save_dir / f"segment_{seg_idx:02d}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"Saved JSON: {save_path}")

    print(f"Segment {seg_idx:02d} done.")
