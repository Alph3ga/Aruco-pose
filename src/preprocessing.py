import cv2
import numpy as np

THRESHOLD = 100.0

def bilateral_filter(img: np.ndarray) -> np.ndarray:
    fimg= cv2.bilateralFilter(img, 9, 40, 30)  # values manually fine-tuned by testing on pictures of aruco tags
    return fimg

def canny_edge(img: np.ndarray) -> np.ndarray:

    img= bilateral_filter(img)

    gradX= cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize= 3)
    gradY= cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize= 3)

    grad= np.sqrt(gradX**2 + gradY**2)
    tanAngle= gradX/gradY

    rows, columns= img.shape
    flagMat= np.zeros((rows, columns))

    for i in range(rows):
        for j in range(columns):
            if grad[i][j]< THRESHOLD:
                continue
            if flagMat[i][j]==0:
                pnts= getSquare(grad, tanAngle, i, j, 0, -1, -1, flagMat)
                if pnts is not None:
                    print(pnts)



def getSquare(grad: np.ndarray[float, float], tanAngle:np.ndarray[float, float], i:int, j:int, size:float, 
              dir:int, line:int, flagMat: np.ndarray[int, int], pnts: list[tuple[int, int]]= []) -> list[tuple[int]]:
    
    if flagMat[i][j]==1:
        if len(pnts)!= 4:
            return None
        return pnts
    
    flagMat[i][j]=1

    if dir==0:
        pnts.append[(i,j)]
    
    currTan= tanAngle[i][j]

    # try and condense this into something shorter plz
    if line==0:
        if dir==1:
            prevTan= tanAngle[i][j-1]
        else:
            prevTan= tanAngle[i][j-1]
    elif line==1:
        if dir==1:
            prevTan= tanAngle[i+1][j+1]
        else:
            prevTan= tanAngle[i-1][j-1]
    elif line==2:
        if dir==1:
            prevTan= tanAngle[i+1][j]
        else:
            prevTan= tanAngle[i-1][j]
    elif line==3:
        if dir==1:
            prevTan= tanAngle[i+1][j-1]
        else:
            prevTan= tanAngle[i-1][j+1]

    if line== -1 or np.abs(currTan-prevTan)-(np.pi/2.0)< 0.09:  # maintaining a straight line, within about 5 degrees
        if currTan<= 0.414 and currTan>= -0.414:  # x axis
            line= 0
            if grad[i, j+1]>= THRESHOLD:
                if dir==0:
                    dir=1
                if dir==1:
                    return getSquare(grad, tanAngle, i, j+1, size+1.0, dir, line, pnts, flagMat)
            if grad[i, j-1]>= THRESHOLD:
                if dir==0:
                    dir=-1
                if dir==-1:
                    return getSquare(grad, tanAngle, i, j-1, size+1.0, dir, line, pnts, flagMat)

        if currTan<= 2.414 and currTan> 0.414:  # x=y line
            line= 1
            if grad[i+1, j+1]>= THRESHOLD:
                if dir==0:
                    dir=1
                if dir==1:
                    return getSquare(grad, tanAngle, i+1, j+1, size+1.414, dir, line, pnts, flagMat)
            if grad[i-1, j-1]>= THRESHOLD:
                if dir==0:
                    dir=-1
                if dir==-1:
                    return getSquare(grad, tanAngle, i-1, j-1, size+1.414, dir, line, pnts, flagMat)
        
        if currTan> 2.414 and currTan< -2.414:  # y axis
            line= 2
            if grad[i+1, j]>= THRESHOLD:
                if dir==0:
                    dir=1
                if dir==1:
                    return getSquare(grad, tanAngle, i+1, j, size+1.0, dir, line, pnts, flagMat)
            if grad[i-1, j]>= THRESHOLD:
                if dir==0:
                    dir=-1
                if dir==-1:
                    return getSquare(grad, tanAngle, i-1, j, size+1.0, dir, line, pnts, flagMat)

        if currTan< -0.414 and currTan> -2.414:  # x=-y line
            line= 3
            if grad[i+1, j-1]>= THRESHOLD:
                if dir==0:
                    dir=1
                if dir==1:
                    return getSquare(grad, tanAngle, i+1, j-1, size+1.414, dir, line, pnts, flagMat)
            if grad[i-1, j+1]>= THRESHOLD:
                if dir==0:
                    dir=-1
                if dir==-1:
                    return getSquare(grad, tanAngle, i-1, j+1, size+1.414, dir, line, pnts, flagMat)
                
    # Code for when the line segment ends
    
    if size< 50.0:
        return None
    
    # see if the perpendicular direction crosses threshold
    if line== 0:
        line=2
        if grad[i+1][j]>= THRESHOLD:
            dir= 1
            pnts.append((i+1,j))
            return getSquare(grad, tanAngle, i+1, j, 0, dir, line, pnts, flagMat)
        if grad[i-1][j]>= THRESHOLD:
            dir= -1
            pnts.append((i-1,j))
            return getSquare(grad, tanAngle, i-1, j, 0, dir, line, pnts, flagMat)
    if line== 1:
        line=3
        if grad[i+1][j-1]>= THRESHOLD:
            dir= 1
            pnts.append((i+1,j-1))
            return getSquare(grad, tanAngle, i+1, j-1, 0, dir, line, pnts, flagMat)
        if grad[i-1][j+1]>= THRESHOLD:
            dir= -1
            pnts.append((i-1,j+1))
            return getSquare(grad, tanAngle, i-1, j+1, 0, dir, line, pnts, flagMat)
    if line== 2:
        line=0
        if grad[i][j+1]>= THRESHOLD:
            dir= 1
            pnts.append((i,j+1))
            return getSquare(grad, tanAngle, i, j+1, 0, dir, line, pnts, flagMat)
        if grad[i-1][j]>= THRESHOLD:
            dir= -1
            pnts.append((i,j-1))
            return getSquare(grad, tanAngle, i, j-1, 0, dir, line, pnts, flagMat)
    if line== 3:
        line=1
        if grad[i+1][j+1]>= THRESHOLD:
            dir= 1
            pnts.append((i+1,j+1))
            return getSquare(grad, tanAngle, i+1, j+1, 0, dir, line, pnts, flagMat)
        if grad[i-1][j-1]>= THRESHOLD:
            dir= -1
            pnts.append((i-1,j-1))
            return getSquare(grad, tanAngle, i-1, j-1, 0, dir, line, pnts, flagMat)
    
    if len(pnts)!=4:
        return None
    return pnts


img= np.ones((100,100))
img= img.astype(np.uint8)*255
cv2.line(img, (10,10), (90,10), (0,0,0))
cv2.line(img, (10,10), (10,90), (0,0,0))
cv2.line(img, (10,90), (90,90), (0,0,0))
cv2.line(img, (90,10), (90,90), (0,0,0))

canny_edge(img)
            