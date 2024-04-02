import copy

import numpy as np
import open3d as o3d

# ---------------读取-------------------#
pcd = o3d.t.io.read_point_cloud("ScanPos002.pcd")
pcd_intensity = pcd.point["intensity"]
pcd_points = pcd.point["positions"]
pcd_intensity = pcd_intensity[:, :].numpy()
pcd_points = pcd_points[:, :].numpy()

# 多项式系数
b0 = 2.41
b1 = 2.27
b2 = -2.42
b3 = 1
c0 = 152.1
c1 = -10.28
c2 = 1.861
c3 = -0.1119
c4 = 2.162e-3
M2 = -4.87e-8

# 中心点坐标
x0 = -307
y0 = -89.99
z0 = 68.615

# distance
R = []
for i in range(pcd_intensity.shape[0]):
    r = np.sqrt(((pcd_points[i][0] - x0) ** 2) + ((pcd_points[i][1] - y0) ** 2) + (
            (pcd_points[i][2] - z0) ** 2))
    R.append(r)
R = np.array(R)

# cos
pcd1 = o3d.io.read_point_cloud("ScanPos002.pcd")
pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(max_nn=30, radius=1))
o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd1, camera_location=np.array([x0, y0, z0]))
pcd_normals = np.array(pcd1.normals)
pcd_incident = copy.deepcopy(pcd_points)
cos_incident = []
for i in range(pcd_incident.shape[0]):
    pcd_incident[i][0] = -1 * pcd_incident[i][0] + x0
    pcd_incident[i][1] = -1 * pcd_incident[i][1] + y0
    pcd_incident[i][2] = -1 * pcd_incident[i][2] + z0
    x = pcd_incident[i, :]
    y = pcd_normals[i, :]
    cos_incident.append(x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))))
cos_incident = np.abs(np.array(cos_incident))

# correction model
for i in range(pcd_incident.shape[0]):
    # f = M2*(b0*np.power(b0*cos_incident[i], 0) + b1*np.power(cos_incident[i], 1) + b2*np.power(cos_incident[i],
    # 2) + np.power(cos_incident[i], 3)) * (c0*np.power(R[i], 0)+c1*np.power(R[i], 1)+c2*np.power(R[i],
    # 2)+c3*np.power(R[i], 3)+c4*np.power(R[i], 4)+np.power(R[i], 5))
    f1 = M2 * (b0 * np.power(b0 * cos_incident[i], 0) + b1 * np.power(cos_incident[i], 1) + b2 * np.power(
        cos_incident[i], 2) + np.power(cos_incident[i], 3))
    f2 = c0 * np.power(R[i], 0) + c1 * np.power(R[i], 1) + c2 * np.power(R[i], 2) + c3 * np.power(R[i],
                                                                                                  3) + c4 * np.power(
        R[i], 4)
    # distance correct
    if R[i] < 20.11:
        pcd_intensity[i][0] = pcd_intensity[i][0] / f2
    else:
        pcd_intensity[i][0] = pcd_intensity[i][0] / (166.3 * np.exp(-0.008 * R[i]))
    # incident angle correct
    pcd_intensity[i][0] = pcd_intensity[i][0] / f1
pcd_intensity = (pcd_intensity - np.min(pcd_intensity)) / (np.max(pcd_intensity) - np.min(pcd_intensity))

# ---------------生成-------------------#
device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
pcd = o3d.t.geometry.PointCloud(device)
pcd.point["positions"] = o3d.core.Tensor(pcd_points, dtype, device)
pcd.point["intensity"] = o3d.core.Tensor(pcd_intensity, dtype, device)
o3d.t.io.write_point_cloud("ScanPos002_corrected.pcd", pcd, write_ascii=True)
