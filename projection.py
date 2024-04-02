import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("ScanPos001_corrected.pcd")
points = np.array(pcd.points)
x = points[:, 0] + 292.27
y = points[:, 1] + 112.08
z = points[:, 2] - 62.86

r = np.sqrt(x**2 + y**2 + z**2)
yaw = -np.arctan2(y, x)
pitch = np.arcsin(z/r)

points[:, 0] = np.cos(yaw) * np.cos(pitch)
points[:, 1] = np.sin(yaw) * np.cos(pitch)
points[:, 2] = np.sin(pitch)

pcd.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("ScanPos001_projection.pcd", pcd)
