import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
import cv2


if __name__ == '__main__':
    FOV_Up = 61
    FOV_Down = 40
    H = 1667
    W = 6000
    x0 = -292.27
    y0 = -112.04
    z0 = 64.12

    pcd = o3d.io.read_point_cloud("ScanPos001 - SINGLESCANS - 221015_135803.pcd")
    points = np.array(pcd.points)
    x = points[:, 0] - x0
    y = points[:, 1] - y0
    z = points[:, 2] - z0

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(z / r)

    fov_up_rad = np.deg2rad(FOV_Up)
    fov_down_rad = np.deg2rad(FOV_Down)

    # projection
    u = 0.5 * (1 + yaw / np.pi) * W
    v = (1 - (pitch + fov_down_rad) / (fov_up_rad + fov_down_rad)) * H

    g = np.zeros((2000, 562), dtype=np.uint8)

    u = np.ceil(u / 3)
    print(np.min(u), np.max(u))
    v = np.ceil(v / 3)
    print(np.min(v), np.max(v))

    for i in range(u.shape[0]):
        e = int(u[i]) - 1
        f = int(v[i]) - 1
        g[e][f] += 1.0

    # plt.imshow(g, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    # cv2.imwrite("test.png", g)
    #
    # plt.figure(1)
    # plt.hist(g)
    # # 显示横轴标签
    # plt.xlabel("x")
    # # 显示纵轴标签
    # plt.ylabel("frequency")
    # plt.show()

    X = g.reshape(-1)
    X1 = X[:562000]
    X2 = X[562000:]

    # Initialize parameters
    mu1_hat, sigma1_hat = np.mean(X1), np.std(X1)
    mu2_hat, sigma2_hat = np.mean(X2), np.std(X2)
    pi1_hat, pi2_hat = len(X1) / len(X), len(X2) / len(X)

    # Perform EM algorithm for 20 epochs
    num_epochs = 30
    log_likelihoods = []

    for epoch in tqdm(range(num_epochs)):
        # E-step: Compute responsibilities
        gamma1 = pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
        gamma2 = pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)
        total = gamma1 + gamma2
        gamma1 /= total
        gamma2 /= total

        # M-step: Update parameters
        mu1_hat = np.sum(gamma1 * X) / np.sum(gamma1)
        mu2_hat = np.sum(gamma2 * X) / np.sum(gamma2)
        sigma1_hat = np.sqrt(np.sum(gamma1 * (X - mu1_hat) ** 2) / np.sum(gamma1))
        sigma2_hat = np.sqrt(np.sum(gamma2 * (X - mu2_hat) ** 2) / np.sum(gamma2))
        pi1_hat = np.mean(gamma1)
        pi2_hat = np.mean(gamma2)

        # Compute log-likelihood
        log_likelihood = np.sum(np.log(pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
                                       + pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)))
        log_likelihoods.append(log_likelihood)

    p_1 = pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat) / (pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat) + pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat))
    p_1 = p_1.reshape((2000, 562))

    for i in range(p_1.shape[0]):
        for j in range(p_1.shape[1]):
            if g[i][j] == 0:
                p_1[i][j] = 0.0

    # plt.imshow(p_1, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    # cv2.imwrite("p.png", p_1)

    c = []
    for i in range(100):
        for j in range(28):
            if p_1[i*20][j*20] >= 0.5:
                c.append([i*20, j*20])
    c = np.array(c) + 1
    c_candidate = []
    for i in tqdm(range(c.shape[0])):
        ra = []
        index = []
        if g[c[i][0]][c[i][1]] == 0:
            print("none")
            continue
        for j in range(u.shape[0]):
            if c[i][0] == u[j] and c[i][1] == v[j]:
                ra.append(r[j])
                index.append(j)
        ra = np.array(ra)
        index = np.array(index)
        a = index[np.argmin(ra, axis=0)]
        c_candidate.append([x[a], y[a], z[a]])
    c_candidate = np.array(c_candidate)

    # ---------------生成-------------------#
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point["positions"] = o3d.core.Tensor(c_candidate, dtype, device)
    o3d.t.io.write_point_cloud("glass.pcd", pcd, write_ascii=True)
