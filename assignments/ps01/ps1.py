import math
import numpy as np
import cv2
import sys

def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image = np.copy(image)
    #temp_image[:, :, 1] = temp_image[:, :, 0] = 0
    return temp_image[:, :, 2]
    #raise NotImplementedError


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    temp_image = np.copy(image)
    #temp_image[:, :, 2] = temp_image[:, :, 0] = 0
    return temp_image[:, :, 1]
    #raise NotImplementedError


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    temp_image = np.copy(image)
    #temp_image[:, :, 1] = temp_image[:, :, 2] = 0
    return temp_image[:, :, 0]
    #raise NotImplementedError


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    temp_image = np.copy(image)
    temp_image[:, :, 1] =  image[:, :, 0] # Copy to blue
    temp_image[:, :, 0] =  image[:, :, 1] # Copy to blue
    return temp_image
    # raise NotImplementedError


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    h, w = shape
    temp_image = np.copy(dst)
    
    src_h, src_w=src.shape
    dst_h, dst_w=dst.shape
    
    src_h_ix_start = (src_h-h)/2
    src_w_ix_start = (src_w-w)/2
    
    dst_h_ix_start = (dst_h-h)/2
    dst_w_ix_start = (dst_w-w)/2   
    
    temp_image  [dst_h_ix_start : dst_h_ix_start + h, dst_w_ix_start : dst_w_ix_start + w]\
           = src[src_h_ix_start : src_h_ix_start + h, src_w_ix_start : src_w_ix_start + w]
    
    return temp_image 
    #raise NotImplementedError


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    temp_image = np.copy(image).astype(np.float64)
    return temp_image.min(), temp_image.max(), temp_image.mean(), temp_image.std()
    #raise NotImplementedError


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    
    temp_image = np.copy(image).astype(np.float64)
    mean_, std_ = temp_image.mean(), temp_image.std()
    temp_image = ((( temp_image[:] - mean_ ) / std_ ) * scale ) + mean_
    
    return temp_image.astype(np.uint8)
    #raise NotImplementedError



def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    temp_image = np.copy(image)
    if shift == 0:
        return temp_image
    temp_image[:, :-shift] = temp_image[:, shift:]
    img_border = cv2.copyMakeBorder(temp_image,0,0,0,shift,cv2.BORDER_REPLICATE)
    temp_image[:,-shift:] = img_border[:,-shift:]
    return temp_image

    #raise NotImplementedError

def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    
    temp_diff  = img1.astype(np.float) - img2.astype(np.float)
    temp_diff -= temp_diff.min()
    if temp_diff.max() != 0:
        temp_diff /= temp_diff.max()
    temp_diff *= 255.
    
    #temp_diff = temp_diff*(255/(255-temp_diff.max()))

    return temp_diff#.astype(np.uint8)
    
    #raise NotImplementedError


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    temp_image = np.copy(image).astype(np.float64)
    h, w, _ = temp_image.shape
    
    noise=np.random.randn( h, w )*sigma
    
    temp_image[:,:,channel] = temp_image[:,:,channel] + noise
    
    return temp_image.astype(np.int16)
    #raise NotImplementedError
