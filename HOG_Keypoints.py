'''
Cameron Husted
Image Stitching - HOG & Keypoints Functions
'''

import cv2
import numpy as np

def getHOG(cornerLocations, image):
    # Vast majority of the credit for this function goes to EECS 504, where they provided a lot of the
    # structure for this HOG code, including parameter initializations.
    win_size = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(win_size, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    stride = (8,8)
    padding = (8,8)
    newLocations = []

    # Gathering all corner locations
    for i in range(cornerLocations.shape[0]):
        newLocations.append((int(cornerLocations[i][0]), int(cornerLocations[i][1])))
    N = len(newLocations)

    descriptors = hog.compute(image, stride, padding, newLocations)
    featureSize = int((((win_size[0] / 8) - 1) * ((win_size[1] / 8) - 1)) * 36)
    HOG_Features = np.reshape(descriptors, (N, featureSize))

    return HOG_Features

def MatchKeypoints(HOG_Features1, HOG_Features2, img1Corners, img2Corners):
    # "Brute-Force Matcher" takes the descriptor of one feature in first set and is matched with
    #       all other features in second set using some distance calculation. And the closest one is returned
    # If "crossCheck" is true, BFMatcher returns only those matches with value(i, j) such
    #       that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = bf.match(HOG_Features1, HOG_Features2)

    matches = sorted(matches, key=lambda x:x.distance)
    img1Matched, img2Matched = [], []
    for match in matches:
        img1Matched.append(img1Corners[match.queryIdx])
        img2Matched.append(img2Corners[match.trainIdx])

    return img1Matched, img2Matched
