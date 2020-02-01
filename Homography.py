'''
Cameron Husted
Image Stitching - Calculating Homography Matrix
'''

import numpy as np

def getHomography(img1_keypoints, img2_keypoints):
    # Calculates Homography Matrix based on given keypoins

    img1_keypoints = np.asarray(img1_keypoints)         # (numCorrespondences, 2)
    img2_keypoints = np.asarray(img2_keypoints)         # (numCorrespondences, 2)

    n = img1_keypoints.shape[0]                         # numCorrespondences (# of points)
    b = np.zeros((2 * n, 1))
    x = np.zeros(n)
    y = np.zeros(n)
    x2 = np.zeros(n)
    y2 = np.zeros(n)
    for i in range(n):                                  # Loops through all keypoints
        x[i] = img1_keypoints[i, 0]                     # extract (x,y) for first image
        y[i] = img1_keypoints[i, 1]
        x2[i] = img2_keypoints[i, 0]                    # extract (x,y) for second image
        y2[i] = img2_keypoints[i, 1]

    A = np.zeros((n * 2, 8))
    for i in range(0, n):                               # create matrix A and b
        ind = 2 * i                                     # looking to skip two rows at a time per loop
        b[ind] = x2[i]
        b[ind + 1] = y2[i]
        A[ind] = np.array([[x[i], y[i], 1, 0, 0, 0, -x2[i] * x[i], -x2[i] * y[i]]])
        A[ind + 1] = np.array([[0, 0, 0, x[i], y[i], 1, -y2[i] * x[i], -y2[i] * y[i]]])

    # Solve via LLS
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    h = np.append(h, 1)                         # attach extra 1
    h = np.reshape(h, (3, 3))                   # and reshape

    Hom = np.linalg.inv(h)                      # invert homography, since we want img2 in perspective of img1
    Hom = Hom / Hom[2, 2]                       # normalize so that last element is still a 1
    return Hom
