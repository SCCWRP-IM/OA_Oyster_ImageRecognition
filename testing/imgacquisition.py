''' This file consists of functions that will deal with image acquisition.
The following functions will obtain the image in digital form for scaling and 
color conversion.
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
Project
------------------------------------------------------------------------------
Create a program that takes the input of an image of oysters set on a petri 
dish. From the analysis, information about the image will be given by computer
vision.
------------------------------------------------------------------------------
Assumptions
------------------------------------------------------------------------------
(1) The size of the petri dish will be constant. (Record radius in units cm)
(2) 
------------------------------------------------------------------------------
Considerations
------------------------------------------------------------------------------
(1) OpenCV and Matplotlib recognize the RGB values differently. OpenCV reads 
the list as [B, R, G], whereas Matplotlib reads the list as [R, G, B].
(2) 
------------------------------------------------------------------------------
Expected Values
------------------------------------------------------------------------------
(1) no. of oysters = 20
(2) sizes of each as 1x20 array --> measure each later
(3) 
'''
# ----------------------------------------------------------------------------
''' 
Name                Description
------------------------------------------------------------------------------
img_orig            Image original
x_scale             Scaling factor for x-axis in 2D-coordinate system
y_scale             Scaling factor for y-axis in 2D-coordinate system
'''

'''
Algorithm
------------------------------------------------------------------------------
(1) Read the image.
(2) Determine length of imaage.
(3) Define new width and height of image.
(4) Compute scaling factor.
(5) Scale image.
(6) Save image.
'''
import cv2
import numpy as np 
''' i did not write that def 
pulled from stack overflow '''

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
''' end (not my work) '''


img = cv2.imread("oysterref.JPG", cv2.IMREAD_GRAYSCALE)
img = image_resize(img, height = 800)
img = cv2.GaussianBlur(img, (5, 5), 0)


sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
canny = cv2.Canny(img, 100, 150)


cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("SobelX", sobelx)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("SobelY", sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Laplacian", laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()



