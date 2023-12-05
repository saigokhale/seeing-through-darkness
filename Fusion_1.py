# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:46:42 2023

@author: Admin
"""
import numpy as np

import cv2

from matplotlib import pyplot as plt

import os

from PIL import Image

import shutil

PATH_T="D:/new2/Thermal_trial_fusion2/"
PATH_V="D:/new2/Visible_trial_fusion2/"
PATH="D:/new2/Fusion_Trial2/"

# shutil.rmtree(PATH)

# os.mkdir(PATH)

# img1 = cv2.imread('/content/clusterii_thermal0051.jpg')
# img2 = cv2.imread('/content/clusterii_visible0051.jpg')

visible_im=os.listdir(PATH_V)
thermal_im=os.listdir(PATH_T)

sorted(visible_im)
sorted(thermal_im)

#FUSION METHOD

def fuse_images(visible_img, thermal_img, weight=0.5):

   # Resize the visible image to the same size as the thermal image
   visible_img = cv2.resize(visible_img, (thermal_img.shape[1], thermal_img.shape[0]))

   # Convert the images to floating point data type
   visible_img = visible_img.astype(np.float32)
   thermal_img = thermal_img.astype(np.float32)

   # Fuse the images using a simple average
   fused_img = weight * visible_img + (1 - weight) * thermal_img

   # Normalize the pixel values and convert back to 8-bit data type
   fused_img = cv2.normalize(fused_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

   return fused_img


#STEP 1: Fusion of visible and thermal image

# keep weight = 0.5 for equal fusion of both visible and thermal 

for i in range(len(thermal_im)):
    
    img1=cv2.imread(PATH_T+thermal_im[i])
    
    img2=cv2.imread(PATH_V+visible_im[i])

    fused = fuse_images(img2, img1, weight=0.5)

    plt.imshow(fused)

    cv2.imwrite(PATH+"trial_{}.jpg".format(thermal_im[i][2:]),fused)


# STEP 2: Create an image with sharpened edges

# Create the laplacian sharpening kernel

    kernel = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
    
    # Apply the sharpening kernel to the image using filter2D
    
    sharpened = cv2.filter2D(fused, -1, kernel)
    
    plt.imshow(sharpened)
    
    # cv2.imwrite(PATH+'sharpened{}.jpg'.format(thermal_im[i][2:]), sharpened)


# STEP 3: Fusion of fused image and image with sharpened edges

#Change weight parameter as per requirement

    fused_edges = fuse_images(fused, sharpened, weight=0.8)
    
    plt.imshow(fused_edges)
    
    cv2.imwrite(PATH+'fused_edges{}.jpg'.format(thermal_im[i][2:]), fused_edges)