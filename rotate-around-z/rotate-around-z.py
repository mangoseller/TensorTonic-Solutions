import numpy as np

def rotate_around_z(points, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    return (R @ np.asarray(points).T).T
