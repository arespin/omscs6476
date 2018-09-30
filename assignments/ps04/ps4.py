"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to
                             [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    return cv2.Sobel(image,ddepth=-1,dx=1,dy=0,scale=0.125, ksize=3)
    #raise NotImplementedError


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    return cv2.Sobel(image,ddepth=-1,dx=0,dy=1,scale=0.125, ksize=3)
    #raise NotImplementedError


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    
    test = False
    show = False
    if test:
        img_a, img_b = shift_0.copy(), shift_r2.copy()
    if show:
        cv2.imshow('img_a', img_a)
        cv2.imshow('img_b', img_b)
        cv2.waitKey(0)
    
    # Compute gradients
    img_A = img_a.copy() 
    img_B = img_b.copy() 
    
    Ix = gradient_x(img_A)
    Iy = gradient_y(img_A)
    # Temporal Derivative
    It = img_B - img_A
    # Method to solve U and V
    posm = [('00', (Ix, Ix)), ('01', (Ix, Iy)), \
            ('10', (Iy, Ix)), ('11', (Iy, Iy))]
    posn = [('0' , (Ix, It)), ('1' , (Iy, It))]
    if k_type == 'uniform':
        kernel = np.ones((k_size,k_size), dtype=np.float32)/(k_size**2)
        m = dict((k, cv2.filter2D(i*j, -1, kernel)) for k, (i,j) in  posm)
        n = dict((k, cv2.filter2D(i*j, -1, kernel)) for k, (i,j) in  posn)
    elif k_type == 'gaussian':
        m = dict((k, cv2.GaussianBlur(i*j, (0,0), sigma)) for k, (i,j) in  posm)
        n = dict((k, cv2.GaussianBlur(i*j, (0,0), sigma)) for k, (i,j) in  posn)
        
    
    det = (m['00']*m['11'] - m['01']*m['10'])
    det[det == 0] = 10**-5
    det = 1.0/det

    U = det*( m['11']) * -n['0'] + det*(-m['10']) * -n['1']
    V = det*(-m['01']) * -n['0'] + det*( m['00']) * -n['1']
    
    return U, V

def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    
    # Apply the kernel and take every second pixel
    img_out = cv2.sepFilter2D(image.copy(), -1, kernel(), kernel())[::2,::2]
    return img_out

def kernel():
    return np.array([1, 4, 6, 4, 1])/16.

def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    # image = yos_img_01
    gpyr = []
    img = image.copy()
    for i in range(levels):
        gpyr.append(img)
        img = reduce_image(img)

    return gpyr


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    # img_list = yos_img_01_g_pyr
    h, w     = img_list[0].shape[0], sum([i.shape[1] for i in img_list])
    template = np.zeros((h, w))
    
    x_from = 0
    for img in img_list:
        hi, wi   = img.shape
        template[0:hi, x_from:(x_from+wi)] = normalize_and_scale(img, scale_range=(0, 255))
        x_from += wi
    
    show = False
    if show:
        cv2.imshow('img_a', template)
        cv2.waitKey(0)
    
    return template


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
        
    h, w = image.shape
    template = np.zeros((h * 2, w * 2))
    template[::2,::2] = image.copy()
    
    img_out = cv2.sepFilter2D(template, -1, kernel(), kernel())*4.0
    
    return img_out


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    
    lpyr = []
    for i in range(len(g_pyr)-1):
        h, w = g_pyr[i].shape
        img  = g_pyr[i] - expand_image(g_pyr[i + 1])[:h, :w]
        lpyr.append(img)
    lpyr.append(g_pyr[-1])

    return lpyr


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    # image, U, V = yos_img_02, u, v
    
    h, w = image.shape
    
    X, Y = [i.astype('float32') for i in np.meshgrid(range(h), range(w), indexing='ij')]
    
    X += V[:h, :w]
    Y += U[:h, :w]
    
    return cv2.remap(image, Y, X, interpolation=interpolation, borderMode=border_mode)
    

def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    
    # img_a, img_b = shift_0, shift_r10
    gpyr_a = gaussian_pyramid(img_a, levels)[::-1]
    gpyr_b = gaussian_pyramid(img_b, levels)[::-1]
    
    U, V = np.zeros(gpyr_a[0].shape), np.zeros(gpyr_b[0].shape)
    
    for t, (a, b) in enumerate(zip(gpyr_a, gpyr_b)):
        
        if t==0:
            warped = gpyr_b[0]
        else:
            warped = warp(b, U, V, interpolation, border_mode)
            
        ui, vi = optic_flow_lk(a, warped, k_size, k_type, sigma)

        h, w = ui.shape
        U = U[:h, :w] + ui

        h, w = vi.shape
        V = V[:h, :w] + vi
        
        if (t+1)<levels:
            U, V = 2. * expand_image(U), 2. * expand_image(V)
        
    
    return U, V