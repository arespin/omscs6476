"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
# https://docs.google.com/document/d/1bsMUYCGK3v6ooYdnmL7aa8Y_JDkyboL9TKJOmWYGcAE/edit
import cv2
import numpy as np


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    
    # detect circles in the image
    #img_in     = tl.copy()
    img_gray   = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    minr, maxr = radii_range[0]-5, radii_range[-1]+10
    circlemat = cv2.HoughCircles(img_gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 50, \
                            param1=20, param2=10, \
                            minRadius=minr, maxRadius=maxr)
    #if circlemat is None:
    #    print 'AAAAAAAAAAAAAAAAAAAAAAAAaaa'
    #    return None
    # get rid of the furthest circles on the x-axis
    #print 50*'*' 
    circlemat = circlemat[0]
    '''
    print 50*'*' 
    print circlemat.shape
    print 50*'*'
    '''
    #print circlemat.shape
    #print circlemat
    while circlemat.shape[0]>3:
        diff_pos = abs(circlemat[:,0]-np.mean(circlemat[:,0])).argmax()
        circlemat = np.delete(circlemat, diff_pos, 0)
    #print circlemat
    
    '''
    # Loop through till we find the correct radius
    circlemat = circlemat[0]
    print circlemat
    # Take steps along the x axis
    for xpos in circlemat[:,0]:
        idx = np.where(np.logical_and(circlemat[:,0]>=xpos-5, circlemat[:,0]<=xpos+5))
        # if we have three circle in the same range, and all the same x axis, we have our traffic light
        if idx[0].shape[0]==3:
            break
    # Take out the correct 3 circles
    circlemat = circlemat[idx].astype(np.int32)
    '''
    #print 50*'-'
    #print circlemat
    #print 50*'-'
    # Sort by height and get the centre coordinate
    circlemat = circlemat[circlemat[:,1].argsort()].astype(np.int64)
    x, y      = circlemat[1,:2].tolist()
    # Use HLS color scheme to find which light is the brightest
    hls        = cv2.cvtColor(img_in, cv2.COLOR_BGR2HLS)
    hls        = cv2.dilate(hls, np.ones((3,3)))
    brightness = np.array([hls[y1, x1][1] for (x1,y1)  in  circlemat[:,:2].tolist()])
    color      = ['red', 'yellow', 'green'][brightness.argmax()]  

    return ((x, y), color)

def get_edges(img):
    edges = cv2.Canny(img,50,400)
    return edges

def get_angle(coords):
    x1, y1, x2, y2 = coords
    angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    return angle


def mask_color(img, color = 'red'):
    
    # define range of colors in HSV
    ranges = { 
      'black'     : (np.array([0, 0, 0]), np.array([179, 50, 100])) ,
      'red'     : (np.array([-20, 100, 100]), np.array([13, 255, 255])) ,
      'dark_red':(np.array([0,45,45]     ), np.array([10,255,255])) ,
      'yellow'  : (np.array([20, 50, 150]  ), np.array([40, 255, 255])) ,
      'green'   : (np.array([50, 50, 150]  ), np.array([70, 255, 255])) ,
      'orange'  : (np.array([10, 50, 50]   ), np.array([20, 255, 255]))
     }

    # Threshold the HSV image to get[13, 255, 255] only relevant colors
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ranges[color][0], ranges[color][1])
    mask = cv2.bitwise_and(img,img, mask= mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    return mask

def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    try:
        #img_in = sign_img.copy() 
        # Find a mask on red 
        mask = mask_color(img_in, color = 'red')
        
        # Get the edges 
        edges=get_edges(mask)
        #HoughlinesP to find lines
        lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 20,minLineLength = 30,maxLineGap = 100)[0].tolist()
        # Lets check if its a red triangle, by comparing the angles to the angles we expect
        actual_angles   = [get_angle(l) for l in lines]
        expected_angles = [0, 60, -60]  
        # Compare each expected angle to make sure we have at least on hit
        if not all([sum([1 for aa in actual_angles if abs(aa-ea)<1])>1  for ea in expected_angles]):
            pass
        # Get the centre - match the angle of the line to a list
        # Start with one side, and then use that as anhor to see which lines with different angle meet one of the corners
        side1 = [[l for l in lines if abs(get_angle(l)-0)<2][0]][0]
        sides = [[l for l in lines if abs(get_angle(l)-ea)<2 and \
                  ((abs(l[0] - side1[0])<4) or (abs(l[2] - side1[2])<4))][0]  for ea in expected_angles]
        # Get center - avg x's and y's
        x, y  = sum([l[0]+l[2] for l in sides])/6, sum([l[1]+l[3] for l in sides])/6
    
        return (x, y)
    except:
        #'Yield sign fails'
        return None

def line_length(line):
    return ((line[2]-line[0])**2 + (line[3]-line[1])**2)**0.5

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    try:
        #img_in = sign_img.copy() 
        #img_in = scene.copy()
        # Find a mask on red 
        mask = mask_color(img_in, color = 'dark_red')
        # distinguish the yield red from the stop sign red
        mask[np.where((img_in[:,:,2]>240))] = 0
        
        # dilate to get rid of noise
        mask = cv2.dilate(mask, np.ones((3,3)))
        # Brighten the mask
        mask[mask>20] = 255 
        # Get the edges 
        edges=get_edges(mask)
        
        #cv2.imshow('img_in', img_in)
        #cv2.imshow('mask', mask)
        #cv2.waitKey(0)
        #HoughlinesP to find lines
        '''
        lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 10,minLineLength = 10,maxLineGap = 1000)[0].tolist()
        # Make sure the lines are not too short or long
        lines = [l for l in lines if ((line_length(l)>10) and (line_length(l)<100))]
        # Drop the horizontal lines; there are a lot of them
        lines = [l for l in lines if abs(get_angle(l) - abs(45))<10]
        # Lets check if its the correct shape, by comparing the angles to the angles we expect
        actual_angles   = [get_angle(l) for l in lines]
        #print(lines)
        #print(actual_angles) 
        expected_angles = [-90, -45, 45]  
        #print all([sum([1 for aa in actual_angles if abs(aa-ea)<4])>0  for ea in expected_angles])
        # Compare each expected angle to make sure we have at least on hit
        if not all([sum([1 for aa in actual_angles if abs(aa-ea)<4])>0  for ea in expected_angles]):
            pass
        # Get the centre lines as 45 and 90 degrees
        #sides = [[l for l in lines if abs(get_angle(l)-ea)<2][0] for ea in expected_angles   
        sides = [l for l in lines if any([abs(get_angle(l)-ea)<2 for ea in expected_angles])]
        # Get center - avg x's and y's
        x, y  = sum([l[0]+l[2] for l in sides])/(2*len(sides)), sum([l[1]+l[3] for l in sides])/(2*len(sides))
        '''
        y, x = tuple(int(np.mean(i)) for i in np.where(mask>250))
        
        return (x, y)
    except:
        #print 'Stop sign fails'
        return None

def diamond_sign_detection(img_in, color_ = 'yellow'):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    try:
        #img_in = sign_img.copy() 
        #color_ = 'orange'
        # Find a mask on red 
        mask = mask_color(img_in.copy(), color = color_)
        # dilate to get rid of noise
        mask = cv2.dilate(mask, np.ones((3,3)))
        # Brighten the mask
        mask[mask>20] = 255 
        # Get the edges 
        edges=get_edges(mask)
        
        #cv2.imshow('img_in', img_in)
        #cv2.imshow('mask', mask)
        #cv2.waitKey(0)
        #HoughlinesP to find lines
        lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 40,minLineLength = 15,maxLineGap = 100)[0].tolist()
        lines
        # Drop the horizontal lines; there are a lot of them
        lines = [l for l in lines if (abs(get_angle(l) -0) - 45)<5]
        # Sort the lines
        lines = np.array(lines)
        for i in range(4):
            lines = lines [lines [:,i].argsort()]
        # Remoev any lines which are too close
        prev_row = [0,0,0,0]
        idx = []
        for row in lines:
            idx.append(sum([abs(r1-r2) for (r1,r2) in zip(row, prev_row)]))
            prev_row = row.copy()
        idx = [i>10 for i in idx]
        lines = lines[idx]
        x, y  = int(lines[:,[0,2]].mean()), int(lines[:,[1,3]].mean())
        '''
        # Lets check if its the correct shape, by comparing the angles to the angles we expect
        actual_angles   = [get_angle(l) for l in lines]
        actual_angles 
        expected_angles = [45]  
        # Compare each expected angle to make sure we have at least on hit
        if not all([sum([1 for aa in actual_angles if abs(aa-abs(ea))<1])>1  for ea in expected_angles]):
            pass
        
        # Get the centre lines as 45 and 90 degrees
        sides = [[l for l in lines if abs(get_angle(l)-ea)<2] for ea in expected_angles]
        # Get center - avg x's and y's
        x, y  = sum([l[0]+l[2] for l in sides])/4, sum([l[1]+l[3] for l in sides])/4
        '''
        
        return (x, y)
    except:
        #print color_ + ' diamond sign fails'
        return None


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    
    return diamond_sign_detection(img_in, color_ = 'yellow')


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    return diamond_sign_detection(img_in, color_ = 'orange')


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    #img_in = sign_img.copy() 
    # img_in = scene.copy()
    # Find a mask on red 
    mask = mask_color(img_in, color = 'red')
    
    # distinguish the do not enter red from the stop sign red
    mask[np.where((img_in[:,:,2]<220) & (img_in[:,:,2]>200))] = 0

    #cv2.imshow('img_in', img_in)
    #cv2.imshow('mask', mask)
    #cv2.imshow('masky', maskyld)
    #cv2.imshow('masks', maskstp)
    #cv2.waitKey(0)
    
    # dilate to get rid of noise
    mask = cv2.dilate(mask, np.ones((3,3)))
    # Brighten the mask
    mask[mask>20] = 255 
    # Get the edges 
    edges=get_edges(mask)
    lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 10,minLineLength = 15,maxLineGap = 100)[0].tolist()
    actual_angles   = [get_angle(l) for l in lines]
    #actual_angles 
    # Check if there are 2 or more straight lines in the red mask
    if sum([1 for a in actual_angles if abs(a)<2])>1 is None:
        return None
    circlemat = cv2.HoughCircles(mask, cv2.cv.CV_HOUGH_GRADIENT, 1, 50, \
                            param1=20, param2=10, \
                            minRadius=20, maxRadius=100)[0]
    if circlemat is None:
        return None
    

    
    # Take the largest circle by radius
    # Sort by height and get the centre coordinate
    circlemat = circlemat[circlemat[:,2].argsort()]
    x, y      = circlemat[0,:2].astype(np.int32).tolist()
    
    return (x, y)


def small_traffic_light_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    try:
        #img_in = sign_img.copy() 
        # Find a mask on red 
        mask = mask_color(img_in, color = 'black')
        
        # dilate to get rid of noise
        mask = cv2.dilate(mask, np.ones((3,3)))
        # Brighten the mask
        mask[mask>20] = 255 
        # Get the edges 
        edges=get_edges(mask)
        
        #cv2.imshow('img_in', img_in)
        #cv2.imshow('mask', mask)
        #cv2.waitKey(0)
        #HoughlinesP to find lines
        lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 30,minLineLength = 30,maxLineGap = 200)[0].tolist()
        # Drop the horizontal lines; there are a lot of them
        lines = [l for l in lines if abs(get_angle(l) -0)>5]
        #print lines
        # Lets check if its the correct shape, by comparing the angles to the angles we expect
        actual_angles   = [get_angle(l) for l in lines]
        #actual_angles 
        #print actual_angles
        expected_angles = [-90]  
        # Compare each expected angle to make sure we have at least on hit
        if not all([sum([1 for aa in actual_angles if abs(aa-ea)<1])>1  for ea in expected_angles]):
            pass
        
        # Get center - avg x's and y's
        x, y  = sum([l[0]+l[2] for l in lines])/(2*(len(lines))), sum([l[1]+l[3] for l in lines])/(2*(len(lines)))
        
        return (x, y)
    except:
        #print 'Stop sign fails'
        return None



def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    
    outdict = {}
    funcpairs = {'yield'        : yield_sign_detection,
                 'stop'         : stop_sign_detection,
                 'warning'      : warning_sign_detection,
                 'traffic_light' : small_traffic_light_detection,
                 'construction' : construction_sign_detection,
                 'no_entry'     : do_not_enter_sign_detection}
    for (name, fn) in funcpairs.items():
        signloc = fn(img_in.copy())
        if signloc is None:
            continue
        outdict[name] = signloc
        
    #print 50*'-'
    
    
    return outdict


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    #img_in = scene.copy()
    dst = cv2.fastNlMeansDenoisingColored(img_in.copy(),None,10,10,7,21)

    return traffic_sign_detection(dst)

    cv2.imshow('img_in', img_in)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    raise NotImplementedError


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    dst = cv2.fastNlMeansDenoisingColored(img_in.copy(),None,10,10,7,21)

    return traffic_sign_detection(dst)
    raise NotImplementedError
