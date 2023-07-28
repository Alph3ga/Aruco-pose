# Aruco-Pose-Estimation
Repository for Aruco pose estimation for Python

## Development and use:
### How to set up:
- You need to have Python installed >=3.11.0
- If you have `GNU make`, run `make init`
- Else, simply run `pip install -r requirements.txt`
- If you want to set up requirements manually, see the *requirements.txt* file

### Testing:
- If you have `GNU make`, run `make test`
- Else, run `python -m unittest tests/tests_unit.py`

## About the methods used:

Aruco Markers have square black borders, which a code encoded within them by black and white squares. They can be easily detected in a variety of lighting conditions. 

For pose detection, using a pin-hole camera model, one needs 3 points to detect complete pose of a pre-calibrated camera \(intrinsic parameters known\), and 6 points for an uncalibrated camera. One tag has four corners, which means if the dimensions of a tag are known, you only need 1 marker to detect the pose of a pre-calibrated camera. Calibration can be done using 2 markers on a board.

Detection of pose from Aruco tags is divided into several subparts:
- Detection of possible square tags
- Detection of the inner encoding of the tags
- Solving a PnP problem to get camera pose

Only the first step has been implemented in this repository yet.

### Detection of possible square tags:
Firstly, the image needs to be processed to remove both noise and texture, as they can interfere in the detection of contours. This is done by a bilateral filter, which only keeps the edges sharp.

The processed image then needs to be converted into a binary valued image, with only zeros and ones. This is done by using a Local Adaptive Thresholding Technique[^1] , which takes into consideration the brightness value of a window around a pixel, marking all 'dark' pixels as 1, and rest as 0.

Now, the Suzuki-Abe algorithm[^2] is used to follow the contours in the binary image and seperate them out. These contours are then simplified into a polygon by downsampling based on distance from the lines of the polygon. The simplified polygons need to be further filtered to ensure that they are convex quadrilaterals, and have enough area. 

The filtered polygons are now candidates to be Aruco Markers.

### Detection of the inner encoding of the tags:
A perspective transformation is used to map the convex quadrilateral to a square. Averaged out values of inner pixels is used to find out the encoding, based on the knowledge of the number of bits encoded (the dictionary from which the marker is generated) in the marker. 

The code is compared to the codes in the dictionary to see if it is valid code. Dictionaries shouldn't be unnecessarily large in order to reduce false positives.

### Solving a PnP problem to get camera pose:
The object to image mapping of the camera can be approximated as a simple linear map using a matrix by assuming a pinhole camera model.
For object coordinates matrix X, image coordinates x, scalar a, and camera matrix P, the transform is given by ax=PX. (1s and 0s are appended to adjust for dimensions). [^3]

Pose estimation consists of solving for the matrix P, or the unknowns within it. P consists of 11 unknowns, 3 transfomations, 3 rotations, and 4 camera parameters. One point yields 2 equation, for the 2 coordinates of the image. This means, when the 4 camera parameters are known, there are 6 unknowns, which need just 3 points to solve for. Since an Aruco Marker has 4 corners, one marker is enough for this.


## References:
- Giuseppe Papari, Nicolai Petkov, ["Edge and line oriented contour detection: State of the art"](https://www.cs.rug.nl/~petkov/publications/2011ivc-contour_survey.pdf), *Image and Vision Computing*, Vol 29, no. 2-3, pp. 79-103 \[Online\]
- B ter Haar Romeny, L Florack, J Koenderink, M Viergever Eds, ["A Review of Nonlinear Diusion"](https://web.cs.hacettepe.edu.tr/~erkut/cmp717.f20/materials/weickert_review.pdf), *Scale-Space Theory in Computer Vision, Lecture Notes in Computer Science*, Vol 1252, Springer, Berlin, pp. 3-28 \[Online\]
- Francisco J. Romero-Ramirez, Rafael Muñoz-Salinas, et al., ["Speeded up detection of squared fiducial markers"](http://andrewd.ces.clemson.edu/courses/cpsc482/papers/RMM18_speededAruco.pdf), *Image and Vision Computing*, Vol 76, pp. 38-47 \[Online\]
[^2]: Satoshi Suzuki, Keiichi Abe, ["Topological Structural Analysis of Digitized Binary Images by Border Following"](https://www.nevis.columbia.edu/~vgenty/public/suzuki_et_al.pdf), *COMPUTER VISION, GRAPHICS, AND IMAGE PROCESSING*, Vol 30, pp 32-46 \[Online\]
[^1]: T.Romen Singh , Sudipta Roy, et al., ["A New Local Adaptive Thresholding Technique in Binarization"](https://arxiv.org/ftp/arxiv/papers/1201/1201.5227.pdf), *IJCSI International Journal of Computer Science Issues*, Vol. 8, Issue 6, No 2, pp. 271-276 \[Online\]
- JianPo Guo et al, ["A precision pose measurement technique based on multi-cooperative logo"](https://iopscience.iop.org/article/10.1088/1742-6596/1607/1/012047/pdf) 2020 Journal of Physics: Conference Series 1607 012047
[^3]: Hristo Hristov, ["The Direct Linear Transform"](https://www.baeldung.com/cs/direct-linear-transform), Baeldung CS \[Online\]
