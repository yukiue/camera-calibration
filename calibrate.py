#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

CSV_FILENAME = sys.argv[1]

d = np.loadtxt(CSV_FILENAME, delimiter=',', skiprows=1)
n = d.shape[0]

# estimate camera-parameters

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

# validate camera-parameters

c = np.append(c, 1)
c = c.reshape(3, 4)
rp = np.empty((n, 2))
for i in range(n):
    p = np.append(d[i, 0:3], 1)
    x = c @ p
    rp[i, 0] = x[0] / x[2]
    rp[i, 1] = x[1] / x[2]
print(d[:, 3:5] - rp[:, :])

fig, ax = plt.subplots()
ax.scatter(d[:, 3], d[:, 4], s=5)
ax.scatter(rp[:, 0], rp[:, 1], s=5)
plt.show()

# separate intrinsic parameters and extrinsic parameters
