import cv2
import numpy as np


def bilateral_filter(img: np.ndarray) -> np.ndarray:
    fimg= cv2.bilateralFilter(img, 9, 40, 30)  # values manually fine-tuned by testing on pictures of aruco tags
    return fimg


def get_binary(img:np.ndarray, windowSize: int= 35, k: float= 0.06) -> np.ndarray:
    """ Returns binarized image, with only values of 0 and 1.
    Uses algorithm laid out by T.Romen Singh and Sudipta Roy in their paper. This uses a window to calculate if a pixel is
    light (0) or dark (1). The values are chosen to aid the contour detection. 
    """
    fimg= img.astype(float)/255.0

    sumImage= np.zeros(fimg.shape)  # calculating the integral sum image, makes calculating window mean a constant time operation
    r, c= sumImage.shape

    sum=0
    for i in range(r):
        sum+= fimg[i][0]
        sumImage[i][0]= sum
    sum=0
    for i in range(c):
        sum+= fimg[0][i]
        sumImage[0][i]= sum

    for i in range(1,r):  # since this itself is O(n^2), amortized calculation becomes O(1)
        for j in range(1,c):
            sumImage[i][j]= fimg[i][j]+ sumImage[i-1][j]+ sumImage[i][j-1]- sumImage[i-1][j-1]

    d= int(windowSize/2 if windowSize%2==0 else (windowSize+1)/2)
    
    for i in range(r):
        for j in range(c):
            di= max(0, i-d)
            dj= max(0, j-d)
            Di= min(i+d-1, r-1)
            Dj= min(j+d-1, c-1)
            mean= ((sumImage[Di][Dj]+ sumImage[di][dj])- (sumImage[di][Dj]+ sumImage[Di][dj]))/(windowSize*windowSize)
            meanDev= fimg[i][j]- mean
            if fimg[i][j]> mean*(1.0+ k*((meanDev/(1.0-meanDev))-1.0)):
                fimg[i][j]= 0.0
            else:
                fimg[i][j]= 1.0

    return fimg
        