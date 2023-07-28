import cv2
import numpy as np
import os

def get_from_camera(camNumber: int= 0, height: int= 800) -> np.ndarray:
    """Captures picture from camera 'camNumber' and returns a numpy array.
    Returns None on error. 
    """
    
    cam= cv2.VideoCapture(camNumber)
    ret, img= cam.read()  # ret is a bool indicating if successful

    if ret:
        h, w, c= img.shape
        img= cv2.resize(img, (800, int(800*w/h)))
        return img
    else:
        return None
    
def get_from_file(path: str, size: tuple[int, int]= None) -> np.ndarray:
    """Gets picture from path and returns a numpy array, optionally resized to size (int, int).
    Returns None on error. 
    """

    if not os.path.isfile(path):  # check if file exists
        return None
    
    img= cv2.imread(path, cv2.IMREAD_COLOR)

    if size is None:
        h, w, c= img.shape
        return cv2.resize(img, (800, int(800*w/h)))
    else:
        return cv2.resize(img, size, interpolation= cv2.INTER_CUBIC)
    
