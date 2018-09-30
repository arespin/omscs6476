"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "output"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in xrange(0, v.shape[0], stride):

        for x in xrange(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """
    
    for l in range(level - 1, -1, -1):
        h, w = pyr[l].shape
        u = (2. * ps4.expand_image(u))[:h, :w]
        v = (2. * ps4.expand_image(v))[:h, :w]
    return u, v

def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 40 #0  # TODO: Select a kernel size
    k_type = "uniform" #""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 40  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma  = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.
    
    def optik_flowlk_write(img0, img1, iname, k_size, k_type, sigma, gks, gksd):
        #Try blurring your images or smoothing your results
        img0 = cv2.GaussianBlur(img0, (gks,gks), gksd)
        img1 = cv2.GaussianBlur(img1, (gks,gks), gksd)
        U, V = ps4.optic_flow_lk(img0, img1, k_size, k_type, sigma)
        # Flow image
        u_v = quiver(U, V, scale=3, stride=10)
        cv2.imwrite(os.path.join(output_dir, iname), u_v)
    
    optik_flowlk_write(img0   = shift_0, 
                       img1   = shift_r10, 
                       iname  = "ps4-1-b-1.png", 
                       k_size = 100, 
                       k_type = "uniform", 
                       sigma  = 0,
                       gks    = 15,
                       gksd   = 3)
    
    optik_flowlk_write(img0   = shift_0, 
                       img1   = shift_r20, 
                       iname  = "ps4-1-b-2.png", 
                       k_size = 125, 
                       k_type = "uniform", 
                       sigma  = 0,
                       gks    = 25,
                       gksd   = 5)
    
    optik_flowlk_write(img0   = shift_0, 
                       img1   = shift_r40, 
                       iname  = "ps4-1-b-3.png", 
                       k_size = 150, 
                       k_type = "uniform", 
                       sigma  = 0,
                       gks    = 35,
                       gksd   = 10)

    #raise NotImplementedError


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
            
    #print yos_img_01.shape
    #print yos_img_02.shape
    levels   = 5  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size   = 0  # TODO: Select a kernel size
    k_type   = "gaussian"  # TODO: Select a kernel type
    sigma    = 20  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation     = cv2.INTER_CUBIC  # You may try different values
    border_mode       = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels   = 5  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size   = 0  # TODO: Select a kernel size
    k_type   = "gaussian"  # TODO: Select a kernel type
    sigma    = 20  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 6  # TODO: Define the number of levels
    k_size = 60  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma  = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    
    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    levels = 5  # TODO: Define the number of levels
    k_size = 20  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma  = 10 # TODO: Select a sigma value if you are using a gaussian kernel
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    #print shift_0
    #print u20
    #print 50*'--'
    u_v = quiver(u20, v20, scale=3, stride=30)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)
    
    levels = 5  # TODO: Define the number of levels
    k_size = 40  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma  = 30 # TODO: Select a sigma value if you are using a gaussian kernel
    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=30)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 5  # TODO: Define the number of levels
    k_size = 40  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma  = 20  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=9, stride=30)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))

def interpolate(img0, img1, levels, k_size, 
                               k_type, sigma, interpolation, border_mode):
    # Hierarchical Lucas-Kanade method forward and backward direction
    ufwd, vfwd = ps4.hierarchical_lk(img0, img1, levels, k_size, 
                               k_type, sigma, interpolation, border_mode)
    ubwd, vbwd = ps4.hierarchical_lk(img1, img0, levels, k_size, 
                               k_type, sigma, interpolation, border_mode)
    
    # Save the flows, including the first which is raw image
    #flowls = [ps4.warp(img0, -0.2*i*u, -0.2*i*v, interpolation, border_mode) for i in range(0,5)]
    flowlsfwd = [img0]
    flowlsbwd = [img1]
    for i in range(1,3):
        flowlsfwd.append(ps4.warp(flowlsfwd[-1], -0.2*ufwd, -0.2*vfwd, interpolation, border_mode))
        flowlsbwd.append(ps4.warp(flowlsbwd[-1], -0.2*ubwd, -0.2*vbwd, interpolation, border_mode))

    #print(flowlsfwd[0].shape, flowlsbwd[0].shape)
    # Stack the image frames
    frames = np.vstack((np.hstack(flowlsfwd),np.hstack(flowlsbwd[::-1])))
    
    return frames

def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.

    levels = 6  # TODO: Define the number of levels
    k_size = 60  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma  = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    
    # get the frames
    frames = interpolate(shift_0, shift_r10, levels, k_size, 
                               k_type, sigma, interpolation, border_mode)
    # Write out
    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-1.png"),
                ps4.normalize_and_scale(frames))


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    mc01 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc01.png'), 0) / 255.
    mc02 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc02.png'), 0) / 255.
    mc03 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 'mc03.png'), 0) / 255.
    
    levels = 8  # TODO: Define the number of levels
    k_size = 20  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma  = 6  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    
    # get the frames part b1
    frames = interpolate(mc01, mc02, levels, k_size, 
                               k_type, sigma, interpolation, border_mode)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-1.png"),
                ps4.normalize_and_scale(frames))
    
    # get the frames part b2
    frames = interpolate(mc02, mc03, levels, k_size, 
                               k_type, sigma, interpolation, border_mode)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-2.png"),
                ps4.normalize_and_scale(frames))
    
def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
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

def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.cv.CV_FOURCC(*'MP4V')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    skip   = 2
    video1 = 'input_videos/racecar.mp4'
    print video1

    image_gen1 = video_frame_generator(video1)
    image1 = image_gen1.next()
    for _ in range(skip):
        image2 = image_gen1.next()

    h1, w1 = image1.shape[:2]

    frame_num = 1
    out_path = "output/optic_flow_racecar.mp4"
    video_out = mp4_video_writer(out_path, (w1, h1), fps = 20)

    frame_ct = 0
    while (image2 is not None) and (frame_ct <300):
        frame_ct += 1
        print "Processing frame {}".format(frame_num)
        
        levels = 5  # TODO: Define the number of levels
        k_size = 50  # TODO: Select a kernel size
        k_type = "gaussian"  # TODO: Select a kernel type
        sigma  = 20 # TODO: Select a sigma value if you are using a gaussian kernel
        interpolation = cv2.INTER_CUBIC  # You may try different values
        border_mode = cv2.BORDER_REFLECT101  # You may try different values
    
        image1bw = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2bw = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        image1bw = ps4.normalize_and_scale(image1bw*1.,  scale_range=(0, 1))
        image2bw = ps4.normalize_and_scale(image2bw*1.,  scale_range=(0, 1))
        
        u20, v20 = ps4.hierarchical_lk(image1bw, image2bw, levels, k_size, k_type,
                                       sigma, interpolation, border_mode)
        u_v = quiver(u20, v20, scale=3, stride=20)
        
        image_out = image1.copy()
        image_out[np.where(u_v>10)] = 255
        if frame_ct==30:
            cv2.imwrite(os.path.join(output_dir, "ps4-6-a-1.png"), image_out)
        if frame_ct==200:
            cv2.imwrite(os.path.join(output_dir, "ps4-6-a-2.png"), image_out)
        
        video_out.write(image_out)
        image1 = image2.copy()
        for _ in range(skip):
            image2 = image_gen1.next()

        frame_num += 1

    video_out.release()


if __name__ == "__main__":
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_6()
