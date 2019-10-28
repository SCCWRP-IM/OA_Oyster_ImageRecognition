import numpy as np
import cv2 as cv
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt

import skimage
from skimage import io
from skimage import feature
from skimage import filters
from skimage import measure, morphology, segmentation, color
import matplotlib.pyplot as plt


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

image = cv.imread("photos/oyster_2.jpg")
image = image_resize(image, height = 800)
imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

cv.imshow("gray", imgray)
cv.waitKey(0)
cv.destroyAllWindows()

imgray = cv.GaussianBlur(imgray, (3,3), 0)
cv.imshow("gray blurred", imgray)
cv.waitKey(0)
cv.destroyAllWindows()

#230 works for oysterref.jpg # assumption: this is the quality of the image we will get
#100 for oyster_test.jpg #bas image quality
ret,thresh = cv.threshold(imgray, 185, 255, cv.THRESH_BINARY_INV) # change 1st number fr shadows of shapes
cv.imshow("threshed",thresh)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite("threshed.jpg", thresh)

threshed = io.imread("threshed.jpg")
io.imshow(threshed)
plt.show()

edges = skimage.feature.canny(threshed, sigma = 7)

io.imshow(edges)
plt.show()

dt = distance_transform_edt(~edges)
io.imshow(dt)
plt.show()

local_max = feature.peak_local_max(dt, indices=False, min_distance=3)
io.imshow(local_max)
plt.show()

peak_idx = feature.peak_local_max(dt, indices = True, min_distance = 3)
print("peak_idx")
print(peak_idx)

plt.plot(peak_idx[:,1], peak_idx[:,0], 'r.')
io.imshow(dt)
plt.show()


markers = measure.label(local_max)
labels = morphology.watershed(-dt, markers)
io.imshow(segmentation.mark_boundaries(image, labels))
plt.show()

io.imshow(color.label2rgb(labels, image=image))
plt.show()



