import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

input_dir = Path("output/single_view/2d_fitted_trajectory")
output_dir = Path("output/single_view/3d_estimated_trajectory")
output_dir.mkdir(parents=True, exist_ok=True)

# 3D mapping parameters
x_start, x_end = 0, 2.45
y_start, y_end = 4.0, 1.2
z_start, z_end = 1.8, 3.05

# Court paint/key area (rectangle)
paint_pts = np.array([
    [0, 0, 0],
    [0, 5.8, 0],
    [4.9, 5.8, 0],
    [4.9, 0, 0],
    [0, 0, 0],      # Closed loop
])

# Hoop and backboard definition
hoop_center = np.array([2.45, 1.2, 3.05])
theta = np.linspace(0, 2*np.pi, 100)
hoop_r = 0.23
hx = hoop_center[0] + hoop_r * np.cos(theta)
hy = hoop_center[1] + hoop_r * np.sin(theta)
hz = np.ones_like(hx) * hoop_center[2]
bb_x = [hoop_center[0] - 0.9, hoop_center[0] + 0.9]
bb_y = [hoop_center[1], hoop_center[1]]
bb_z = [hoop_center[2], hoop_center[2]]

# Multiple camera views
views = [
    {"elev": 25, "azim": 120, "title": "Isometric View"},
    {"elev": 10, "azim": 50, "title": "Low Angle (Side)"},
    {"elev": 60, "azim": 180, "title": "High Angle (Behind)"},
    {"elev": 40, "azim": 270, "title": "Corner View"},
]

for idx in range(1, 38):  # For segment_01.json to segment_37.json
    in_path = input_dir / f"segment_{idx:02d}.json"
    out_path = output_dir / f"segment_{idx:02d}.json"
    fig_path = output_dir / f"segment_{idx:02d}_views.png"
    if not in_path.exists():
        print(f"Warning: {in_path} does not exist, skip.")
        continue

    # 1. Read 2D pixel sequence from JSON file
    with open(in_path) as f:
        data = json.load(f)
    pts_2d = np.array(data["trajectory"])  # (N,2) pixel coordinates

    # 2. Proportional mapping for each coordinate to 3D
    x1, x0 = pts_2d[0,0], pts_2d[-1,0]
    y1, y0 = pts_2d[0,1], pts_2d[-1,1]
    xs = x_start + (pts_2d[:,0] - x0) / (x1 - x0) * (x_end - x_start)
    ys = y_start + (pts_2d[:,0] - x0) / (x1 - x0) * (y_end - y_start)
    zs = z_start + (pts_2d[:,1] - y0) / (y1 - y0) * (z_end - z_start)
    traj_3d = np.stack([xs, ys, zs], axis=1)

    # 3. Save 3D trajectory to output directory, as {"trajectory": ...}
    with open(out_path, "w") as f:
        json.dump({"trajectory": traj_3d.tolist()}, f)
    print(f"Saved: {out_path}")

    # 4. Visualize with four different camera angles and save as a single image
    fig = plt.figure(figsize=(18, 4))
    for i, v in enumerate(views):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        # Plot only the actual trajectory, no jump!
        ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color='blue', marker='o', label='Trajectory')
        # Start point
        ax.scatter(traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2], color='green', s=80, label='Start')
        # End point
        ax.scatter(traj_3d[-1, 0], traj_3d[-1, 1], traj_3d[-1, 2], color='red', s=80, label='End')
        # Paint/key area
        ax.plot(paint_pts[:,0], paint_pts[:,1], paint_pts[:,2], color='purple', linewidth=2, label='Paint/key')
        # Backboard
        bb_pts = np.array([
        [1.0, 0.3, 2.8],
        [3.9, 0.3, 2.8],
        [3.9, 0.3, 3.8],
        [1.0, 0.3, 3.8],
        [1.0, 0.3, 2.8]  # close the loop
        ])
        ax.plot(bb_pts[:,0], bb_pts[:,1], bb_pts[:,2], color='black', linewidth=5, label='Backboard')
        
        # Hoop as a hollow orange circle (parallel to xy-plane)
        hoop_center = np.array([2.45, 1.2, 3.05])
        hoop_r = 0.5
        theta = np.linspace(0, 2*np.pi, 100)
        hx = hoop_center[0] + hoop_r * np.cos(theta)
        hy = hoop_center[1] + hoop_r * np.sin(theta)
        hz = np.ones_like(theta) * hoop_center[2]
        ax.plot(hx, hy, hz, color='orange', linewidth=3, label='Hoop')
        
        # Set axis limits and labels
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-0.5, 6.5)
        ax.set_zlim(0, 3.5)
        ax.set_xlabel('X (up)')
        ax.set_ylabel('Y (left)')
        ax.set_zlabel('Height (m)')
        ax.set_title(v["title"])
        ax.view_init(elev=v["elev"], azim=v["azim"])
        if i == 0:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved visualization: {fig_path}")

print("All segments processed.")
