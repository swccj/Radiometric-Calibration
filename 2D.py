import open3d as o3d
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
import seaborn as sns
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告


FOV_Up = 73
FOV_Down = 40

pcd = o3d.io.read_point_cloud("ScanPos001_corrected.pcd")
points = np.array(pcd.points)
x = points[:, 0] + 292.27
y = points[:, 1] + 112.08
z = points[:, 2] - 62.86

r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
yaw = -np.arctan2(y, x)
pitch = np.arcsin(z / r)

# 原点变换
fov_up_rad = (FOV_Up / 180) * np.pi
fov_down_rad = (FOV_Down / 180) * np.pi

# normalization and scale
normalized_pitch = (fov_up_rad - pitch) / (FOV_Up + FOV_Down) * 60000
normalized_yaw = (yaw + np.pi) / (2 * np.pi) * 300000

u = np.ceil(normalized_pitch / 3)
print(np.min(u))
v = np.ceil(normalized_yaw / 3)
print(np.min(v))

g = np.zeros((20000, 100000))
# g = np.zeros((15873, 80000))
for i in range(u.shape[0]):
    x = int(u[i]) - 1
    y = int(v[i]) - 1
    g[x][y] += 1.0
print(np.max(g))
print(np.sum(g))
g = g.reshape(-1)

X = g
X1 = g[:1000000000]
X2 = g[1000000000:]

# Initialize parameters
mu1_hat, sigma1_hat = np.mean(X1), np.std(X1)
mu2_hat, sigma2_hat = np.mean(X2), np.std(X2)
pi1_hat, pi2_hat = len(X1) / len(X), len(X2) / len(X)

# Perform EM algorithm for 20 epochs
num_epochs = 20
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

    print(pi1_hat, pi2_hat, mu1_hat, sigma1_hat, mu2_hat, sigma2_hat)
