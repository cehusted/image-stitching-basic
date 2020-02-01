# image-stitching-basic

### Included Files:
"Stitch Images.py" - Main File
* 'webcam = 1' allows you to take two images from laptop webcam. Otherwise, loads images from file __(Note: When using webcam, press 'q' key to take first image, press 'w' to take second image. Webcam closes automatically after pressing 'w'.)__
* 'numCorrespondences' can be adjusted to change how many keypoints the algorithm uses to match points between images
* Lines 32-33 contain image filepaths for loading from disk

"HCorners.py" - Functions for computing __Harris Corners__ and performing __Non-Max Suppression__ on corners  
"HOG_Keypoints.py" - Functions for getting __HOG Feature Vectors__ and for __Matching Keypoints__ between images  
"Homography.py" - Computes __Homography Matrix__ based on given matched keypoints  
"Crop_Stitched.py" - Removes the black space that resulted from warping the second image to fit it to the first

### This script will save a few images to disk:
* _CornerResponse1/2.png_ - B&W Images showing the response from the __Harris Corners__
* _Stitched Image.jpg_ - Raw stitched image
* _Stitched Image Cropped.jpg_ - Stitched image after cropping

## Example Stitched Images
From Disk - Raw
![Stitched Image](https://raw.githubusercontent.com/cehusted/image-stitching-basic/master/Samples/Stitched_Image4.jpg)
From Disk - Cropped
![Cropped Stitched Image](https://raw.githubusercontent.com/cehusted/image-stitching-basic/master/Samples/Stitched_Image4_Cropped.jpg)
Via Webcam - Raw
![Stitched Image](https://raw.githubusercontent.com/cehusted/image-stitching-basic/master/Samples/Webcam.jpg)
Webcam - Cropped
![Cropped Stitched Image](https://raw.githubusercontent.com/cehusted/image-stitching-basic/master/Samples/Webcam_Cropped.jpg)

