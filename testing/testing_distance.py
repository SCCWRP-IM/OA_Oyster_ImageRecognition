import numpy as np
import cv2
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt

'''
import skimage
from skimage import io
from skimage import feature
from skimage import filters
import matplotlib.pyplot as plt
'''


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

im = cv2.imread('photos/2019_08_08_0060.jpg')
im = image_resize(im, height = 800)

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", imgray)
cv2.waitKey(0)
cv2.destroyAllWindows()

imgray = cv2.GaussianBlur(imgray, (3,3), 0)
cv2.imshow("gray blurred", imgray)
cv2.waitKey(0)
cv2.destroyAllWindows()
#thresh is bad
#230 works for oysterref.jpg # assumption: this is the quality of the image we will get
#100 for oyster_test.jpg #bas image quality
ret,thresh = cv2.threshold(imgray, 185, 255, cv2.THRESH_BINARY_INV) # change 1st number fr shadows of shapes
# canny is irrelevant here so let's ignore it despite its use in size_test.py
# imcanny = cv2.Canny(imgray, 50, 100)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# the oysters are at the following indicies: 169, 189 For the photo oyster_test.jpg


cv2.imshow("threshed", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


for i, cont in enumerate(contours):
    #if i > 7:
    #    break
    if cv2.contourArea(cont) < 100:
        continue
    else:
        cv2.drawContours(im,contours[i],-1,(0,255,0),3)
        cv2.imshow("draw indiv contours", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        points = np.ndarray.tolist(contours[i])
        #change range arg
        p = [[] for i in range(len(contours[i]))]
    
        for j, coord in enumerate(points):
            print(j, coord)
            p[j] = points[j][0]
        
        hull = ConvexHull(p)
    
        _ = convex_hull_plot_2d(hull)
    
        # from the points, does not use hull
        # we want to loop through the points contained in p for each contour. 
        # p is a set of points that form the contour so the contours contains a set of p per contour drawn
        # compute distance
        D = pdist(p)
        # input into an array
        D = squareform(D);
        # find max distance and which points this corresponds to
        N, [I_row, I_col] = nanmax(D), unravel_index( argmax(D), D.shape )
        
        #draw line in original image (we will use this for the bounding boxes)
        cv2.line(im, tuple(p[I_row]), tuple(p[I_col]), (0, 0, 255), 2)
        
cv2.imshow("draw indiv contours", im)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
#change index of contours
points = np.ndarray.tolist(contours[test])
#change range arg
p = [[] for i in range(len(contours[test]))]

for i, coord in enumerate(points):
    print(i, coord)
    p[i] = points[i][0]
    
hull = ConvexHull(p)

_ = convex_hull_plot_2d(hull)

# from the points, does not use hull
# we want to loop through the points contained in p for each contour. 
# p is a set of points that form the contour so the contours contains a set of p per contour drawn
# compute distance
D = pdist(p)
# input into an array
D = squareform(D);
# find max distance and which points this corresponds to
N, [I_row, I_col] = nanmax(D), unravel_index( argmax(D), D.shape )

#draw line in original image (we will use this for the bounding boxes)
cv2.line(im, tuple(p[I_row]), tuple(p[I_col]), (0, 0, 255), 2)
cv2.imshow("draw indiv contours", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
