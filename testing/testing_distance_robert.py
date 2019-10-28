import numpy as np
import cv2
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt

import skimage
from skimage import io
from skimage import feature
from skimage import filters
import matplotlib.pyplot as plt

def get_size(p, theta=range(0,91)): #input is set of points for ONE contour
    size = [[] for lw in range(0,91)] #init size for lw vectors at each theta
    sz = None
    for angle in theta: 
        #convert theta to radians
        th = angle * np.pi /180
        #compute rotation matrix
        rot = np.array([
                [np.cos(th), np.sin(th)],
                [(-1)*np.sin(th), np.cos(th)],
                ])
        #obtain distances by multiplying pts_{n by 2} and rot (rotation matrix)
        dists = np.dot(p, rot) # n by 2 matrix
        length = [[] for l in range(len(p))] # init lengths list
        width = [[] for l in range(len(p))] # init widths list
        for l_row in range(0, len(p)):
            length[l_row] = dists[l_row][0] # grab all "lengths"
        for w_row in range(0, len(p)):
            width[w_row] = dists[w_row][1] # grab all "widths"
        l = max(length) - min(length) # length distance between pts
        w = max(width) - min(width) # width distance between pts
        size[angle] = [l, w] # l w vector for each theta
	if sz != np.amax(size, axis=0): # Satisfaction of this condition implies that the value of sz has changed
            sz = np.amax(size, axis=0) # maximum length width for size of object
	    max_length_index = length.index(max(length)) # in the array p, this is the index where we find one of the points along the contour that defines the length
	    min_length_index = length.index(min(length)) # in the array p, this is the index where we find one of the points along the contour that defines the length
	    max_width_index = length.index(max(width)) # in the array p, this is the index where we find one of the points along the contour that defines the width
	    min_width_index = length.index(min(width)) # in the array p, this is the index where we find one of the points along the contour that defines the width
	    max_length_coord = tuple(p[max_length_index]) # These are for the sake of drawing the lines later
	    min_length_coord = tuple(p[min_length_index])
	    max_width_coord = tuple(p[max_width_index])
	    min_width_coord = tuple(p[min_width_index])
	    length = sz[0]
            width = sz[1]
    return(length, width, size, sz, max_length_coord, min_length_coord, max_width_coord, min_width_coord)

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
'''
 Note: if you run into the the following error: 
     AttributeError: 'NoneType' object has no attribute 'shape'
 for the image_resize function, then explicitly call the full file path.
'''
im = cv2.imread('P:/PartTimers/ZaibQuraishi/ImageCountProject/photos/oysterref.JPG')
im = image_resize(im, height = 800)

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

#cv2.imshow("gray", imgray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

imgray = cv2.GaussianBlur(imgray, (3,3), 0)
#cv2.imshow("gray blurred", imgray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#thresh is bad
#230 works for oysterref.jpg # assumption: this is the quality of the image we will get
#100 for oyster_test.jpg #bas image quality
ret,thresh = cv2.threshold(imgray, 230, 255, cv2.THRESH_BINARY_INV) # change 1st number fr shadows of shapes
# canny is irrelevant here so let's ignore it despite its use in size_test.py
# imcanny = cv2.Canny(imgray, 50, 100)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# the oysters are at the following indicies: 169, 189 For the photo oyster_test.jpg


#cv2.imshow("threshed", thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


for i, cont in enumerate(contours):
    #if i > 7:
    #    break
    
    if cv2.contourArea(cont) < 100:
        continue
    else:
        # for testing purposes
        #if (len(cont) == 115):
        if (len(cont) == 142):
        # end for testing purposes
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
        # N is the max length in unit pixels
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
