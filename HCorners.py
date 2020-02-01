'''
Cameron Husted
Image Stitching - Harris Corners & Non-Max Suppression
'''

import numpy as np
from scipy import signal
import cv2

def HarrisCorners(Gx, Gy, window_half_size, threshold, imgNum):
    # Calculates Harris Corners based off of Sobel-filtered images

    points = []
    cornerResponse = np.zeros(Gx.shape)                     # (600, 800)
    window = np.ones(((2 * window_half_size) + 1, (2 * window_half_size) + 1))

    convX = signal.convolve2d(np.multiply(Gx, Gx), window, mode='same')
    convY = signal.convolve2d(np.multiply(Gy, Gy), window, mode='same')
    convXY = signal.convolve2d(np.multiply(Gx, Gy), window, mode='same')

    for i in range(Gx.shape[0]):
        for j in range(Gx.shape[1]):
            h11 = convX[i,j]                                # sum of squared elements in window of Gx
            h22 = convY[i,j]                                # sum of squared elements in window of Gy
            h_cross = convXY[i,j]

            H = np.array([[h11, h_cross], [h_cross, h22]])  # Structure tensor
            cornerResponse[i,j] = np.min(np.linalg.eigvals(H))

            if cornerResponse[i,j] > threshold:
                points.append((cornerResponse[i,j], i, j))
            else:
                cornerResponse[i, j] = 0

    sortedPoints = [(i, j) for _, i, j in sorted(points, reverse=True)]
    cornerLocations = np.asarray(sortedPoints)              # (480,000, 2)
    cv2.imwrite("CornerResponse{}.png".format(imgNum), cornerResponse / 255)

    return cornerResponse, cornerLocations

def NonMaxSuppression(corner_response, radius):
    # Finds peak response in each area, suppresses all other responses within close radius

    r_map1 = np.copy(corner_response)
    data_max = np.zeros(r_map1.shape)
    ind = np.nonzero(r_map1)

    for n in range(len(ind[0])):
        i = ind[0][n]
        j = ind[1][n]
        frame = r_map1[max((i - radius), 0):min((i + radius), r_map1.shape[0]),
                max((j - radius), 0):min((j + radius), r_map1.shape[1])]
        if r_map1[i, j] < np.max(frame):  # not local max
            data_max[i, j] = 0
        elif np.max(data_max[
                    max((i - radius), 0):min((i + radius), data_max.shape[0]),
                    max((j - radius), 0):min((j + radius), data_max.shape[1])]) > 0:
            # Tie, and already as a max t
            data_max[i, j] = 0
        else:
            data_max[i, j] = r_map1[i, j]

    col_ind, row_ind = np.nonzero(data_max)

    corners = []
    for i in range(len(row_ind)):
        corners.append((row_ind[i], col_ind[i]))
    corners = tuple(corners)

    return np.array(corners)