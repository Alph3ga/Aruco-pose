import cv2
import numpy as np
from scipy import ndimage


def bilateral_filter(img: np.ndarray) -> np.ndarray:
    fimg= cv2.bilateralFilter(img, 9, 40, 30)  # values manually fine-tuned by testing on pictures of aruco tags
    return fimg


def approx_poly(contour: list[np.ndarray[int]], start: int=0, end: int= None, epsilon: float= 5.0) -> list[np.ndarray[int]]:  # Ramer–Douglas–Peucker algorithm for simplifying contour
    # recursively downsamples/removes points on the basis of their distance from a fitting straight line
    # lower epsilon seems a tighter fit, less points removed

    dmax= 0
    index= 0
    if end is None:
        end= len(contour)-1

    for i in range(start+1, end):
        if not (contour[start]-contour[end]).any():
            dist= np.hypot(contour[start][0]-contour[i][0], contour[start][1]-contour[i][1])
        else:
            dist= np.linalg.norm(np.cross(contour[end]- contour[start], contour[start]- contour[i]))/np.linalg.norm(contour[end]- contour[start])
        if dist> dmax:
            dist= dmax
            index= i
    
    if dmax> epsilon:
        partA= approx_poly(contour, start= start, end= index)
        partB= approx_poly(contour, start= index, end= end)

        result= [*partA[:len(partA)-1], *partB[1:len(partB)]]
    else:
        result= [contour[start], contour[end]]
    
    return result


def get_binary(img:np.ndarray, windowSize: int= 35, k: float= 0.06) -> np.ndarray:
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
                fimg[i][j]= 1.0
            else:
                fimg[i][j]= 0.0

    return fimg


def extract_contours(img: np.ndarray) -> list[np.ndarray[int]]:  # modified Moore's neighbor tracking algorithm

    dirList= [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1)]  # clockwise starting from the left neighbor

    encountered_ids= []
    image= np.pad(img, pad_width=1)

    closed_contours= []

    rows, columns= img.shape

    for i in range(rows):
        for j in range(columns):
                label= image[i][j]
                if label== 0: # or label in encountered_ids:  # non boundary pixel or already found
                    continue
                encountered_ids.append(label)

                prevDir= dir= 0
                y= i
                x= j

                contour= []
                contour.append([y-1,x-1])

                while True:
                    dir= (dir+1)%8
                    if dir== prevDir:
                        break
                    dy, dx= dirList[dir]
                    cy= y+dy
                    cx= x+dx

                    if image[cy][cx]== label:
                        contour.append([cy-1,cx-1])
                        dir= (dir+ 4)% 8
                        prevDir= dir
                        x, y= cx, cy

                    if cy== i and cx== j:
                        break
                
                if contour[0]== contour[-1]:
                    closed_contours.append(np.array(contour))
    return closed_contours


def extract_contourss(img: np.ndarray) -> list[np.ndarray[int]]:  # modified Moore's neighbor tracking algorithm

    dirList= [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1)]  # clockwise starting from the left neighbor

    encountered_ids= []
    image= np.pad(img, pad_width=1)

    closed_contours= []

    rows, columns= img.shape

    NBD= 1
    for i in range(rows):
        LNBD= 0
        for j in range(columns):
                label= image[i][j]
                if label== 0: # or label in encountered_ids:  # non boundary pixel or already found
                    continue
                if label==1 and image[i][j-1]==0:
                    pFrom= (i, j-1)
                elif label>=1 and image[i][j+1]==0:
                    pFrom= (i, j+1)
                    LNBD= label
                else:
                    if not label== 1:
                        LNBD= label
                

                prevDir= dir= 0
                y= i
                x= j

                contour= []
                contour.append([y-1,x-1])

                while True:
                    dir= (dir+1)%8
                    if dir== prevDir:
                        break
                    dy, dx= dirList[dir]
                    cy= y+dy
                    cx= x+dx

                    if image[cy][cx]== label:
                        contour.append([cy-1,cx-1])
                        dir= (dir+ 4)% 8
                        prevDir= dir
                        x, y= cx, cy

                    if cy== i and cx== j:
                        break
                
                if contour[0]== contour[-1]:
                    closed_contours.append(np.array(contour))
    return closed_contours



img= cv2.imread("tests/test_images/test4.jpg", cv2.IMREAD_GRAYSCALE)
h, w= img.shape
img= cv2.resize(img, (int(800*w/h), 800))
img= bilateral_filter(img)
img= get_binary(img, windowSize=35, k= 0.07)


img= (img*255).astype(np.uint8)
cv2.imshow("image",img)
cv2.waitKey(0)