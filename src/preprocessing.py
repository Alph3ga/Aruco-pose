import time
import cv2
import numpy as np
from scipy import ndimage


def bilateral_filter(img: np.ndarray) -> np.ndarray:
    fimg= cv2.bilateralFilter(img, 9, 40, 30)  # values manually fine-tuned by testing on pictures of aruco tags
    return fimg


def approx_poly(contour: list[np.ndarray[int]], start: int=0, end: int= None, epsilon: float= 3.0, iter: int= 0) -> list[np.ndarray[int]]:  # Ramer–Douglas–Peucker algorithm for simplifying contour
    # recursively downsamples/removes points on the basis of their distance from a fitting straight line
    # lower epsilon seems a tighter fit, less points removed

    if iter>2:
        return None

    dmax= 0
    index= 0
    if end is None:
        end= len(contour)-2

    line= contour[start]-contour[end]
    slope= float(line[0])/(line[1]+0.000001)
    c= slope*contour[start][1]- contour[start][0]
    norm= 1+ slope*slope
    for i in range(start+1, end):
        if not (contour[start]-contour[end]).any():
            dist= np.hypot(contour[start][0]-contour[i][0], contour[start][1]-contour[i][1])
        else:
            dist= np.abs(contour[i][0]-(slope*contour[i][1])+c)/norm
        if dist> dmax:
            dmax= dist
            index= i
    
    if dmax> epsilon:
        partA= approx_poly(contour, start= start, end= index, iter= iter+1)
        partB= approx_poly(contour, start= index, end= end, iter= iter+1)

        if partA is None or partB is None:
            return None

        result= [*partA[:len(partA)-1], *partB[0:len(partB)]]
    else:
        result= [contour[start], contour[end]]
    
    return result


def approx_polys(contours):
    res= []
    for cnts in contours:
        approxContour= approx_poly(cnts)
        if approxContour is None:
            continue
        res.append(approxContour)
    return res


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
                fimg[i][j]= 0.0
            else:
                fimg[i][j]= 1.0

    return fimg



def extract_contours(img: np.ndarray) -> list[np.ndarray[int]]:  # modified Suzuki-Abe algorithm

    dirList= [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1)]  # clockwise starting from the left neighbor

    # [1 2 3]
    # [0 x 4]
    # [7 6 5]

    image= np.pad(img, pad_width=1)

    closed_contours= []

    rows, columns= img.shape

    NBD= 1
    for i in range(rows):
        LNBD= 1
        for j in range(columns):

            # step 1
            if image[i][j]== 0: 
                continue
            if image[i][j]==1 and image[i][j-1]==0:
                pFrom= (i, j-1)
                NBD+= 1
                dirFrom= 0
            elif image[i][j]>=1 and image[i][j+1]==0:
                pFrom= (i, j+1)
                NBD+= 1
                dirFrom= 4
                LNBD= image[i][j]
            else:
                if not image[i][j]== 1:
                    LNBD= image[i][j] if image[i][j]> 0 else -image[i][j]
                    continue

            contour= []
            
            # step 2
            # skipping impelenting border hierarchy for now

            # step 3

            y= i
            x= j
            prevDir= dirFrom
            while True:
                dirFrom= (dirFrom+1)%8  # search clockwise in the neighbourhood
                if dirFrom== prevDir:
                    image[i][j]= -NBD
                    break
                dy, dx= dirList[dirFrom]
                cy, cx= y+dy, x+dx

                if image[cy][cx]==0:  # look for a non zero pixel
                    continue
                
                pFrom= (cy, cx)
                pStart= (y, x)
                break
            if image[i][j]== -NBD:
                LNBD= NBD
                continue

            prevDir= dir= dirFrom
            contour.append([y,x])

            while True:
                dir= (dir-1)%8  # now search neighbourhood counterclockwise for actual contour following
                #if dir== prevDir:
                    #break
                dy, dx= dirList[dir]
                cy= y+dy
                cx= x+dx

                if image[cy][cx]== 0:  # look for non zero
                    continue


                    
                if image[y][x+1]==0:
                    if (prevDir-dir)%8 > (prevDir-4)%8:
                        image[y][x]= -NBD
                if image[y][x]!= -NBD and image[cy][cx]==1:
                    image[y][x]= NBD

                contour.append([cy,cx])

                if (cy, cx)== pStart and (y, x)== pFrom:  # reached the start point from the starting direction
                    if image[y][x]!=1:
                        LNBD= NBD
                    break
                
                y, x= cy, cx
                dir= (dir+ 4)% 8
                prevDir= dir

                #simg= (img*255).astype(np.uint8)
                #simg= cv2.cvtColor(simg, cv2.COLOR_GRAY2BGR)
                #cv2.line(simg, (x,y), (x+1,y+1), (0, 0, 255), 5)
                #cv2.imshow("hello", simg) # DEBUG, REMOVE
                #print(f"at {y},{x}")
                #cv2.waitKey(10)

            #print(contour)
            if contour[0]== contour[-1] and len(contour)> 50:  # 27 dec 1:06:33 5962
                closed_contours.append(np.array(contour))
                #print(f"{len(closed_contours)} contours found so far", end="  :  ") # DEBUG, REMOVE
    return closed_contours




def filter_poly(polygons: list[np.ndarray]):
    filtered= []
    for poly in polygons:
        if len(poly)!=5:
            continue
        if not is_convex_polygon(poly[0:4]):
            continue
        area= np.linalg.norm(np.cross(poly[1]-poly[0], poly[2]-poly[0]))
        if area< 400.0:
            continue
        iden= False
        for i in filtered:
            iden= identical_polys(i, poly)
        if iden:
            continue
        filtered.append(poly)
    return filtered

def identical_polys(polyA: list[np.ndarray], polyB: list[np.ndarray], epsilon: float= 50.0)-> bool:
    dist= 0
    for i in range(4):
        diff= polyA[i]- polyB[i]
        dist+= np.abs(diff[0])+np.abs(diff[1])
    if dist<= epsilon:
        return True
    return False


TWO_PI = 2 * np.pi

def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.
    """
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = np.arctan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = np.arctan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -np.pi:
                angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
            elif angle > np.pi:
                angle -= TWO_PI
                if angle> 1.3 or angle< -1.3:
                    return False
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon
        
start= time.time()

image= cv2.imread("tests/test_images/test4.jpg", cv2.IMREAD_COLOR)
h, w, c= image.shape
image= cv2.resize(image, (int(800*w/h), 800))
img= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img= bilateral_filter(img)
img= get_binary(img, windowSize=35, k= 0.07)
cnts= extract_contours(img)
print(len(cnts), " contours found")
cnts= approx_polys(cnts)
cnts= filter_poly(cnts)
print(len(cnts), "acceptable contours found")
end= time.time()



img= (img*255).astype(np.uint8)

for cnt in cnts:
    cv2.line(image, cnt[0][::-1], cnt[1][::-1], (0, 255, 0), 1)
    cv2.line(image, cnt[1][::-1], cnt[2][::-1], (0, 255, 0), 1)
    cv2.line(image, cnt[2][::-1], cnt[3][::-1], (0, 255, 0), 1)
    cv2.line(image, cnt[3][::-1], cnt[0][::-1], (0, 255, 0), 1)

cv2.imshow("image",image)
#print(end-start, "seconds taken")
cv2.waitKey(0)