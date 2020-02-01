'''
Cameron Husted
Image Stitching - Main File
'''

from HCorners import HarrisCorners, NonMaxSuppression
from HOG_Keypoints import getHOG, MatchKeypoints
from Crop_Stitched import cropStitched
from Homography import getHomography
from scipy import signal
import numpy as np
import cv2

if __name__ == "__main__":
    webcam = 0
    numCorrespondences = 50

    if webcam:
        camera = cv2.VideoCapture(0)
        while True:
            (_, frame) = camera.read()
            cv2.imshow('Webcam', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                img1 = frame
            if key == ord('y'):
                img2 = frame
                cv2.destroyAllWindows()
                break
        camera.release()
    else:
        img1 = cv2.imread("Image5_Left.png")            # (600, 800, 3)
        img2 = cv2.imread("Image5_Right.png")           # (600, 800, 3)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)      # (600, 800)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)      # (600, 800)

    sobelHorz = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelVert = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    print(" --- Performing filtering operations...")
    Horz1 = signal.convolve2d(gray1, sobelHorz, mode='same')
    Horz2 = signal.convolve2d(gray2, sobelHorz, mode='same')
    Vert1 = signal.convolve2d(gray1, sobelVert, mode='same')
    Vert2 = signal.convolve2d(gray2, sobelVert, mode='same')

    print(" --- Calculating Harris Corners...")
    cornerResponse1, cornerLocations1 = HarrisCorners(Horz1, Vert1, 7, 0.1, 1)     # Takes about 20 seconds
    cornerResponse2, cornerLocations2 = HarrisCorners(Horz2, Vert2, 7, 0.1, 2)

    print(" --- Performing Non-Max Suppression...")
    suppCornerLocations1 = NonMaxSuppression(cornerResponse1, 2)
    suppCornerLocations2 = NonMaxSuppression(cornerResponse2, 2)

    print(" --- Getting HOG feature vectors for each corner...")
    HOG_Features1 = getHOG(suppCornerLocations1, gray1)
    HOG_Features2 = getHOG(suppCornerLocations2, gray2)

    print(" --- Matching feature vectors between images...")
    img1Points, img2Points = MatchKeypoints(HOG_Features1, HOG_Features2, suppCornerLocations1, suppCornerLocations2)

    print(" --- Calculating Homography...")
    H = getHomography(img1Points[1:numCorrespondences + 1], img2Points[1:numCorrespondences + 1])

    print(" --- Stitching images together...")
    stitched = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img2.shape[0]))     # Use homography on img2
    stitched[:img1.shape[0], :img1.shape[1]] = img1                                             # Append img1 to img2
    cv2.imwrite("Stitched Image.jpg", stitched)

    print(" --- Cropping Stitched Image...")
    cropped = cropStitched(stitched)
    cv2.imwrite("Stitched Image Cropped.jpg", cropped)

    print("Complete!")