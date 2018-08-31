#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 20:22:18 2018

@author: darragh
"""

import numpy as np
import cv2
import os, sys
os.chdir('/home/darragh/omscs6476/ps01/')

from ps1 import *

img_filename = ' southafricaflagface.png'
img = cv2.imread(img_filename)

    # # 2b
    img1_green = extract_green(img1)
    assert len(img1_green.shape) == 2, "The monochrome image must be a 2D array"
    cv2.imwrite('output/ps1-2-b-1.png', img1_green)
    
    # # 2c
    img1_red = extract_red(img1)
    assert len(img1_red.shape) == 2, "The monochrome image must be a 2D array"
    cv2.imwrite('output/ps1-2-c-1.png', img1_red)