import cv2
import numpy as np
from scipy import ndimage


def bilateral_filter(img: np.ndarray) -> np.ndarray:
    fimg= cv2.bilateralFilter(img, 9, 40, 30)  # values manually fine-tuned by testing on pictures of aruco tags
    return fimg


def contour_label(image):  # contour following algorithm Suzuki Abe
    # Copy the input image to avoid modifying the original
    segmented_image = np.copy(image)

    # Define the connectivity matrix for 8-connected neighbors
    connectivity = np.array([[0, 0, 0],
                             [0, 1, 2],
                             [4, 8, 16]], dtype=np.uint8)

    # Iterate over the image pixels
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            # Skip background pixels
            if segmented_image[y, x] == 0:
                continue

            # Perform line tracing
            label = 1  # Start a new label
            neighbor_labels = []

            while True:
                # Set the current pixel label
                segmented_image[y, x] = label

                # Find the neighbors of the current pixel
                neighbors = connectivity * (segmented_image[max(0, y-1):y+2, max(0, x-1):x+2] > 0)

                # Find the indices of the non-zero neighbors
                neighbor_indices = np.argwhere(neighbors > 0)

                # Check if there are any non-zero neighbors
                if len(neighbor_indices) == 0:
                    break

                # Get the first non-zero neighbor
                ny, nx = neighbor_indices[0]

                # Update the current pixel coordinates
                y += ny - 1
                x += nx - 1

                # Get the label of the non-zero neighbor
                neighbor_label = neighbors[ny, nx]

                # Check if the neighbor has already been labeled
                if neighbor_label > 1:
                    # Assign the label to the current pixel
                    segmented_image[y, x] = neighbor_label

                    # Add the neighbor label to the list
                    neighbor_labels.append(neighbor_label)

                # Check if line tracing is complete
                if (neighbors == 0).all():
                    break

            # Assign the minimum neighbor label to the connected component
            if len(neighbor_labels) > 0:
                segmented_image[segmented_image == label] = min(neighbor_labels)

    return segmented_image


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


def get_binary(img:np.ndarray, threshold: int= 0.7) -> np.ndarray:
    fimg= img.astype(float)/255.0

    res= fimg>(np.ones(fimg.shape)*threshold)

    return res.astype(np.uint8)


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

    for i in range(rows):
        LNBD= 1
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



def sobel_filter(img: np.ndarray) -> tuple[np.ndarray, np.ndarray] :

    gX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)+ 0.0001
    gY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.hypot(gX, gY)

    tanAngle= gY/gX

    magnitude = (magnitude / float(np.max(magnitude)))  # normalize the array

    return magnitude, tanAngle

def non_maximum_suppression(magnitude, tanAngle):

    suppressed = np.zeros(magnitude.shape)

    for i in range(1, magnitude.shape[0]- 1):
        for j in range(1, magnitude.shape[1]- 1):
            q = 1
            r = 1

            # Angle 0
            if (-0.414 <= tanAngle[i, j] < 0.414):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # Angle 45
            elif (0.414 <= tanAngle[i, j] < 2.414):
                q = magnitude[i+1, j+1]
                r = magnitude[i-1, j-1]
            # Angle 90
            elif tanAngle[i,j]<= -2.414 or tanAngle[i,j]> 2.414:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # Angle 135
            elif (-2.414<= tanAngle[i, j] < -0.414):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed

def double_thresholding(image, low_threshold, high_threshold):

    strongY, strongX = np.where(image >= high_threshold)
    weakY, weakX = np.where((image >= low_threshold) & (image < high_threshold))

    strong = np.zeros(image.shape)
    weak = np.zeros(image.shape)

    strong[strongY, strongX] = 1.0
    weak[weakY, weakX] = 0.5

    return strong, weak

def edge_tracking(strong, weak):
    r, c = strong.shape
    for i in range(1, r-1):
        for j in range(1, c-1):
            if weak[i, j] == 0.5:
                if np.max(strong[i-1:i+2, j-1:j+2])== 1:
                    strong[i, j] = 1
                    weak[i, j] = 0
                else:
                    weak[i, j] = 0

    return strong

def canny_edge_detection(img: np.ndarray, low_threshold: float= 0.4, high_threshold: float= 0.6):

    # Sobel filtering
    magnitude, gradient = sobel_filter(img)

    # Non-maximum suppression
    suppressed = non_maximum_suppression(magnitude, gradient)

    # Double thresholding
    strong, weak = double_thresholding(suppressed, low_threshold, high_threshold)

    # Edge tracking by hysteresis
    edges = edge_tracking(strong, weak)

    return edges

img= cv2.imread("tests/test_images/test1.jpg", cv2.IMREAD_GRAYSCALE)
h, w= img.shape
img= cv2.resize(img, (int(800*w/h), 800))
img= bilateral_filter(img)
img= canny_edge_detection(img)
#img= contour_label(img)
cnts= extract_contours(img)
print(f"{len(cnts)} contours found")
quad= []
for cnt in cnts:
    if len(cnt)<50:
        continue
    cnt= approx_poly(cnt)
    #if len(cnt)==4:
    quad.append(cnt)
print(f"{len(quad)} quads found:\n{quad}")


#img= (img*255).astype(np.uint8)
#cv2.imshow("image",img)
#cv2.waitKey(0)