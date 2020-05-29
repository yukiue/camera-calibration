#!/usr/bin/env python3

import numpy as np
import sys

CSV_FILENAME = sys.argv[1]

d = np.loadtxt(CSV_FILENAME, delimiter=',', skiprows=1)
n = d.shape[0]

# AC = R

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
    a[i * 2 + 1, 9] = -p[0] * p[4]
    a[i * 2 + 1, 10] = -p[0] * p[4]
    r[i * 2] = p[3]
    r[i * 2 + 1] = p[4]
