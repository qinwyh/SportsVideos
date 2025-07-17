import json
import numpy as np
import matplotlib.pyplot as plt

# 1. Define 3D start (release) and end (hoop)
release = np.array([0.0, 4, 1.8])
hoop = np.array([2.45, 1.2, 3.05])

# 2. Load original 2D points, set z=0
with open("output/2d_fitted_trajectory/segment_01.json") as f:
    data = json.load(f)
    orig_pts_2d = np.array(data["original_points"])
N = orig_pts_2d.shape[0]
orig_pts_3d = np.hstack([orig_pts_2d, np.zeros((N, 1))])

# 3. Define shot plane basis: 
#    u_hat: horizontal direction (release to hoop, xy-plane), v_hat: z axis
vec_u = hoop[:2] - release[:2]
len_u = np.linalg.norm(vec_u)
u_hat = np.array([vec_u[0]/len_u, vec_u[1]/len_u, 0])
v_hat = np.array([0, 0, 1])

# 4. For each point: get relative vector q, then s = proj on u_hat, t = proj on v_hat
plane_coords = []
for Q in orig_pts_3d:
    q = Q - release
    s = np.dot(q, u_hat)   # distance along shot direction
    t = np.dot(q, v_hat)   # height (z)
    plane_coords.append([s, t])
plane_coords = np.array(plane_coords)  # shape: (N,2)

# 5. Linearly scale plane_coords so first point at (release->[0,2.3]), last at (hoop->[L,3.05])
#   where L is shot distance in xy-plane
s_start, t_start = 0, 2.3    # desired start in plane
s_end, t_end = len_u, 3.05   # desired end in plane

# Current (s, t) in plane
ss = plane_coords[:, 0]
ts = plane_coords[:, 1]
s_norm = (ss - ss[0]) / (ss[-1] - ss[0])
t_norm = (ts - ts[0]) / (ts[-1] - ts[0])

scaled_ss = s_start + s_norm * (s_end - s_start)
scaled_ts = t_start + t_norm * (t_end - t_start)

# 6. Map back to 3D using plane basis
traj_3d_proj = np.array([
    release + s * u_hat + t * v_hat
    for s, t in zip(scaled_ss, scaled_ts)
])
print(traj_3d_proj)

# 7. Visualization
# fig = plt.figure(figsize=(9,7))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(traj_3d_proj[:,0], traj_3d_proj[:,1], traj_3d_proj[:,2], 'o-', color='b', label='Projected trajectory')
# ax.scatter(*release, color='g', s=80, label='Release [0, 4.6, 2.3]')
# ax.scatter(*hoop, color='r', s=80, label='Hoop [2.45, 1.2, 3.05]')
# ax.set_xlabel('Court X (up)')
# ax.set_ylabel('Court Y (left)')
# ax.set_zlabel('Height (m)')
# ax.set_xlim(0, 15)
# ax.set_ylim(0, 14)
# ax.set_zlim(0, 3.5)
# ax.legend()
# ax.set_title("3D Trajectory (Strictly on Shot Plane)")
# plt.tight_layout()
# plt.show()

# 8. (Optional) Save
with open("output/2d_fitted_trajectory/segment_01_projected_shot_plane_3d.json", "w") as f:
    json.dump(traj_3d_proj.tolist(), f)
print("Saved 3D trajectory strictly on shot plane.")