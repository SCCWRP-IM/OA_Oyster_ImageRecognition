import numpy as np
import cv2 as cv
#from scipy.spatial import ConvexHull, convex_hull_plot_2d

from numpy import random, nanmax, nanmin, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt

import time

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
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
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

im = cv.imread('../cropped_photos/oyster_standard.jpg')
im = image_resize(im, height = 800)

imgray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
'''
cv.imshow("gray", imgray)
cv.waitKey(0)
cv.destroyAllWindows()
'''
imgray = cv.GaussianBlur(imgray, (3,3), 0)
'''
cv.imshow("gray blurred", imgray)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#thresh is bad
#230 works for oysterref.jpg # assumption: this is the quality of the image we will get
#100 for oyster_test.jpg #bas image quality
ret,thresh = cv.threshold(imgray, 185, 255, cv.THRESH_BINARY_INV) # change 1st number fr shadows of shapes
# canny is irrelevant here so let's ignore it despite its use in size_test.py
# imcanny = cv.Canny(imgray, 50, 100)
contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# the oysters are at the following indicies: 169, 189 For the photo oyster_test.jpg

cv.imshow("threshed", thresh)
cv.waitKey(0)
cv.destroyAllWindows()

contours, h = cv.findContours(thresh.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for c in contours:
    accuracy = 0.01 * cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, accuracy, True)
    cv.drawContours(im, [approx], 0, (0,0,255), 2)
    cv.imshow('Approx Poly DP', im)

cv.waitKey(0)
cv.destroyAllWindows()







