#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import math

CSV_FILENAME = sys.argv[1]

d = np.loadtxt(CSV_FILENAME, delimiter=',', skiprows=1)
n = d.shape[0]

# -- estimate camera-parameters --

a = np.zeros((n * 2, 11))
r = np.zeros((n * 2))

for i in range(n):
    p = d[i]
    a[i * 2, 0] = p[0]
    a[i * 2, 1] = p[1]
    a[i * 2, 2] = p[2]
    a[i * 2, 3] = 1
    a[i * 2, 8] = -p[0] * p[3]
    a[i * 2, 9] = -p[1] * p[3]
    a[i * 2, 10] = -p[2] * p[3]
    a[i * 2 + 1, 4] = p[0]
    a[i * 2 + 1, 5] = p[1]
    a[i * 2 + 1, 6] = p[2]
    a[i * 2 + 1, 7] = 1
    a[i * 2 + 1, 8] = -p[0] * p[4]
    a[i * 2 + 1, 9] = -p[1] * p[4]
    a[i * 2 + 1, 10] = -p[2] * p[4]
    r[i * 2] = p[3]
    r[i * 2 + 1] = p[4]

c = np.linalg.inv(a.T @ a) @ a.T @ r

# -- validate camera-parameters --

c = np.append(c, 1)
c = c.reshape(3, 4)
rp = np.empty((n, 2))
for i in range(n):
    p = np.append(d[i, 0:3], 1)
    x = c @ p
    rp[i, 0] = x[0] / x[2]
    rp[i, 1] = x[1] / x[2]
print(d[:, 3:5] - rp[:, :])

# fig, ax = plt.subplots()
# ax.scatter(d[:, 3], d[:, 4], s=5)
# ax.scatter(rp[:, 0], rp[:, 1], s=5)
# plt.show()

# -- separate intrinsic parameters and extrinsic parameters --

print(f'c: {c}')

# - step 1 -

abs_r3 = abs(math.sqrt(c[2, 0]**2 + c[2, 1]**2 + c[2, 2]**2))

print(abs_r3)

c = c * (1.0 / abs_r3)

print(f'normailized c: {c}')

print(math.sqrt(c[2, 0]**2 + c[2, 1]**2 + c[2, 2]**2))

# - step 2 -

r3 = c[2, :3]

tz = c[2, 3]

print(tz)

# - step 3 -

c1 = c[0, :3]
c2 = c[1, :3]
c3 = c[2, :3]

u0 = np.dot(c1, c3)
v0 = np.dot(c2, c3)

# - step 4 -

cross_c1_c3 = np.cross(c1, c3)
cross_c2_c3 = np.cross(c2, c3)

cos_theta = -1 * np.dot(cross_c1_c3, cross_c2_c3) / (
    np.linalg.norm(cross_c1_c3) * np.linalg.norm(cross_c2_c3))

# - step 5 -

sin_theta = math.sqrt(1 - cos_theta**2)
alpha_u = np.linalg.norm(cross_c1_c3) * sin_theta
alpha_v = np.linalg.norm(cross_c2_c3) * sin_theta

# - step 6 -

r2 = (sin_theta / alpha_v) * (c2 - v0 * r3)

ty = (sin_theta / alpha_v) * (c[1, 3] - v0 * tz)

# - step 7 -

cot_theta = cos_theta / sin_theta
r1 = (1 / alpha_u) * (c1 + alpha_u * cot_theta * r2.T - u0 * r3)
tx = (1 / alpha_u) * (c[0, 3] + alpha_u * cot_theta * ty - u0 * tz)
