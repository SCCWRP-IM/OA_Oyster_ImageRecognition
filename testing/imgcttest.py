''' Referencing Source Code from 
https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
'''

from scipy.spatial import distance as dist
from imutils import perspective 
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

''' The package "imutils" will be heavily used. The package containes functions
for the ease of basic image processing operation (translation, rotation, 
resizing, skeletonization, and using/displaying Matplotlib images with
OpenCV and Python).
'''

''' In manipulating the acquired image, we most likely want to adjust the 
resolution to blur the image with a grayscale type image. This grayscale image
is also called the 8-bit color format image. 
Source: https://www.geeksforgeeks.org/digital-image-processing-basics/
'''


def midpoint(pointA, pointB):
    return ( (pointA[0] + pointB[0]) * 0.5, (pointA[1] + pointB[1]) * 0.5)

''' Scaling the Image
Source: https://www.geeksforgeeks.org/image-processing-without-opencv-python/
'''
import matplotlib.image as img
import numpy as npy
'''
The following body of code (1) reads the image, (2) determines the length of 
the image, (3) defines the new width and height of image, (4) computes the 
scaling factor, (4) scale the image, (5) saeve image after scaling.
'''
m = img.imread("oysterref.jpg")
w, h = m.shape[:2]
xNew = int(w*1/2)
yNew = int(h*1/2)
xScale = xNew/(w-1)
yScale = yNew/(h-1)

''' The four parameters are [alpha, B, G, B] values.
'''
newImage = npy.zeros([xNew, yNew, 4])

for i in range(xNew-1):
    for j in range(yNew-1):
        newImage[i + 1, j + 1] = m[1 + int(i / xScale), 1 + int(j / yScale)]
        
img.imsave('scaledref.jpg', newImage)


''' New Body of Code
'''

import cv2
import numpy as np



