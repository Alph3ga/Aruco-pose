import numpy as np


def extract_contours(img: np.ndarray) -> list[np.ndarray]:  # modified Suzuki-Abe algorithm
    """ Return a list of contours, each of which are an numpy ndarray of shape (n,2), where n is the length of contour. 
     This is implemented using a modified Suzuki-Abe Algorithm, specifically, skipping the border heirarchy,
     and appending each complete contour to a list. 

     The directions used are as such:
     [1 2 3]
     [0 x 4]
     [7 6 5]
       """

    dirList= [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1)]  # clockwise starting from the left neighbor

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
            # skipping impelenting border hierarchy 

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
                dy, dx= dirList[dir]
                cy= y+dy
                cx= x+dx

                if image[cy][cx]== 0:  # look for non zero
                    continue
                    
                if image[y][x+1]==0:  # this checks if this is a zero pixel examined in previous step
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

            if contour[0]== contour[-1] and len(contour)> 50:  # implemented a check for minumum length here
                closed_contours.append(np.array(contour))
    return closed_contours

def approx_poly(contour: list[np.ndarray[int]], start: int=0, end: int= None, epsilon: float= 3.0, iter: int= 0) -> list[np.ndarray[int]]:
    """ Returns an approximated polygon contour using Ramer-Douglas-Peucker algorithm for simplifying contour. 
     Recursively downsamples/removes points on the basis of their distance from a fitting straight line. 
     Lower epsilon means a tighter fit, and less points removed. 
     """

    if iter>2:
        return None

    dmax= 0
    index= 0
    if end is None:
        end= len(contour)-2

    # the basic formula for a line and distance of a line from a point is used
    # certain values are pre-calculated to decrease calculations

    line= contour[start]-contour[end]
    slope= float(line[0])/(line[1]+0.000001)  # preventing faulty division
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
    """ Helper function to approximate an entire list of contours. """
    res= []
    for cnts in contours:
        approxContour= approx_poly(cnts)
        if approxContour is None:
            continue
        res.append(approxContour)
    return res


def filter_poly(polygons: list[np.ndarray]):
    """ Filters the list for convex quadrilaterals with large enough area"""
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
    """ Returns true if the points of two polygons are close enough, by manhattan distance. 
    Does not work if they are out of order.
    """
    dist= 0
    for i in range(4):
        diff= polyA[i]- polyB[i]
        dist+= np.abs(diff[0])+np.abs(diff[1])
    if dist<= epsilon:
        return True
    return False


def is_convex_polygon(polygon):
    """Return True if the polynomial is non-self-intersecting, and has interior angles between 0 and 180 degrees. 

    The signed changes of the direction angles from one side to the next side must be all positive or
    all negative, and their sum must equal plus-or-minus 360 degrees.
    """

    TWO_PI = 2 * np.pi
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            return False
        
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = np.arctan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        
        for ndx, newpoint in enumerate(polygon):
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
                if orientation * angle <= 0.0: 
                    return False

            angle_sum += angle

        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon