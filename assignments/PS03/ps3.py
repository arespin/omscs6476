"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    
    return np.linalg.norm(np.array(p0)-np.array(p1))

    raise NotImplementedError

# Find the two farthest points
def two_way_combos(c):
    return [(c[i],c[j]) for i in range(len(c)) for j in range(i+1, len(c))]
# Get the rectangle area around two points
def pair_area(p1, p2):
    return abs(p1[0]-p2[0])*abs(p1[1]-p2[1])


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    
    # image = advert.copy()
    h, w  = image.shape[:2]
    corners = [(0,0), (0, h-1), (w-1, 0), (w-1, h-1)]
    
    return corners

    raise NotImplementedError


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    test = False
    show = False
    if test:
        imgin = scene.copy()
    if show:
        cv2.imshow('scene', imgin)
        cv2.imshow('crnr', crnrmsk)
        cv2.waitKey(0)
        
    imgin = image.copy()
    # Smooth images
    # https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    imgin    = cv2.filter2D(imgin, -1,  0.05 * np.ones((5,5)))
    imgin    = cv2.GaussianBlur(imgin, (3,3), 0)
    # Convert to gray
    img      = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    # Get the corner pixels and only take the higher values
    crnrmsk  = cv2.cornerHarris(img, 8,7,0.04)
    crnrmsk *= (255.0/crnrmsk.max())
    #crnrmsk        = cv2.dilate(crnrmsk, np.ones((6,6))/36 )
    crnrmsk[crnrmsk<50] = 0
    # Make the corner positions into a 2d numpy array
    crnpos = np.array([(i,j) for (i,j) in zip(*np.where(crnrmsk>0))], dtype = np.float32)
    # define criteria and apply kmeans()
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    criteria          = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center  = cv2.kmeans(crnpos,4,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Make list of tuples
    center            = np.fliplr(center).astype(np.int32)
    col0              = np.argsort(center[:,0])
    leftpos           = center[col0[:2]]
    rightpos          = center[col0[2:]]
    tl,bl             = [tuple(l) for l in leftpos[np.argsort(leftpos[:,1])]]
    tr,br             = [tuple(l) for l in rightpos[np.argsort(rightpos[:,1])]]
    
    '''
    #tlidx             = np.intersect1d(np.argsort(center[:,1])[:2], np.argsort(center[:,0])[:2]).ravel()[0]
    #bridx             = np.intersect1d(np.argsort(center[:,1])[2:], np.argsort(center[:,0])[2:]).ravel()[0]
    #tridx             = np.intersect1d(np.argsort(center[:,1])[:2], np.argsort(center[:,0])[2:]).ravel()[0]
    #blidx             = np.intersect1d(np.argsort(center[:,1])[2:], np.argsort(center[:,0])[:2]).ravel()[0]
    center             = [tuple(l) for l in center.tolist()]
    '''
    # [top-left, bottom-left, top-right, bottom-right]
    corners = [tl, bl, tr, br]
    
    return corners # perimeter
    #raise NotImplementedError


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    test = False
    show = False
    if test:
        img = scene.copy()
    if show:
        cv2.imshow('scene', imgin)
        cv2.imshow('crnr', crnrmsk)
        cv2.waitKey(0)
    # # [top-left, bottom-left, top-right, bottom-right] to [top-left, bottom-left,bottom-right, top-right]
    lmarkers = [markers[i] for i in [0, 1, 3, 2, 0]]
    # Rotate through markers
    img = image.copy()
    for p1, p2 in zip(lmarkers[:-1], lmarkers[1:]):
        cv2.line(img, p1, p2, (255), thickness=thickness)
    
    return img
    #raise NotImplementedError

def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    # imageA, imageB, homography = advert, scene, homography
    show = False
    if show:
        cv2.imshow('dst', dst)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        
    imgA         = imageA.copy()
    imgB         = imageB.copy()
    
    h, w, c      = imgB.shape
    H            = np.linalg.inv(homography)
    out          = imgB.copy()
    
    # warp positions
    idxy,idxx    = np.indices((h,w),dtype = np.float32)
    homg_idx     = np.array([idxx.flatten(), idxy.flatten(), np.ones_like(idxx).flatten()])
    mapidx       = H.dot(homg_idx)
    mx, my       = (m.reshape(h, w).astype(np.float32) for m in mapidx[:-1]/mapidx[-1])
    
    # remap this according to warp positions
    dst          = cv2.remap(imgA, mx, my, cv2.INTER_LINEAR)
    # Create mask to overwrite image
    mask         = cv2.remap(np.ones(imgA.shape), mx, my, cv2.INTER_LINEAR)
    # Overwrite with mask
    out[mask==1] = dst[mask==1]

    return out


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    # dst_points = markers
    A = []
    for (x,y), (u,v) in zip(src_points, dst_points):
        #print x,y,u,v
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])


    A = np.matrix(A, dtype=np.float)
    B = np.array(dst_points).flatten()
    af = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.append(np.array(af).flatten(), 1).reshape(3, 3)

    # raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    #video = None
    #filename = video

    vid = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while vid.isOpened():
        ret, frame = vid.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    vid.release()
    yield None
    #raise NotImplementedError
