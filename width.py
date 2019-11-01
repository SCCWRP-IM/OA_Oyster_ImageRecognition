#####################################################################
#                                                                   #
#   THIS IS A SCRIPT TO DEBUG THE WIDTH MEASUREMENT OF THE OYSTERS. #
#                                                                   #
#####################################################################

# import necessary libraries
import numpy as np
import cv2 as cv
from pandas import Series, DataFrame, isnull
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from numpy import random, nanmax, nanmin, argmax, unravel_index
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt
import time
import os
import sys
import re

# Read in the image
im = cv.imread('photos/IMG_9823-resized.jpg')

# Grayscale it
imgray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)

# blur it to reduce noise, limits unnecessary contours from getting detected
imgray = cv.GaussianBlur(imgray, (1,1), 0)

# Threshold it to complete black and white
ret,thresh = cv.threshold(imgray, 185, 255, cv.THRESH_BINARY_INV) # change 1st number fr shadows of shapes

# to see what the code is getting contours from
cv.imwrite("output_images/threshed.jpg", thresh)

# Now the code actually grabs the contours
contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
print("Detected %s contours" % len(contours))

# NOTE When I read in this image and manipulated it with these settings, the contours that were oysters were...
oyster_contours = [58,80,86,93,109,113,123,144,152,158,166,192,198,203,209,215,228]



'''
GET THE CONTOUR POINTS
'''

# set the value of the variable i to change which contour you are dealing with.
i = 58

# Draw the contour on the image
cv.drawContours(im,contours[i],-1,(0,255,0),3)

# contours are just lists of lists of coordinate points 
# (on a basic simple level. I think they are technically np arrays or something)
# Anyways, here we are extracting the list of coordinate points that define each contour to then analyze it
points = np.ndarray.tolist(contours[i])

#change range arg
p = [[] for k in range(len(contours[i]))]

# point is a list of lists of lists.
# we need only a list of lists
for j, coord in enumerate(points):
    #print(j, coord)
    p[j] = points[j][0]    
del points

p = np.array(p)


'''
BEGIN CODE FOR GETTING THE LENGTH
'''

# we want to loop through the points contained in p for each contour. 
# p is a set of points that form the contour so the contours contains a set of p per contour drawn
# compute distance
D = pdist(p)
# input into an array
D = squareform(D);
# find max distance and which points this corresponds to
pixellength = round(nanmax(D), 2)
# called I_row and I_col since it is grabbing the "ij" location in a matrix where the max occurred.
# the row number where it occurs represents one of the indices in the original points array where one of the points on the contour lies
# the column number would be the point on the opposite side of the contour
# L_row, and L_col since these indices correspond with coordinate points that give us the length
[L_row, L_col] = unravel_index( argmax(D), D.shape )

min_length_coord = tuple(p[L_row])
max_length_coord = tuple(p[L_col])
length_coords = [min_length_coord, max_length_coord]
length_vector = np.array(max_length_coord) - np.array(min_length_coord)
unit_length_vector = length_vector / norm(length_vector)      

# draw the line that is being used to measure length, so we get a visual of if it is doing what it is supposed to be doing or not.
cv.line(im, min_length_coord, max_length_coord, (0,255,0))
cv.imwrite('output_images/debug1.jpg', im)


'''
BEGIN CODE FOR GETTING THE WIDTH
'''

# all_vecs will be an list of vectors that are all the combinations of vectors that pass over the contour area
all_vecs = []
coordinates = []
for i in range(0, len(p) - 1):
    for j in range(i + 1, len(p)):
        all_vecs.append(np.array(p[i]) - np.array(p[j]))
        coordinates.append([tuple(p[i]), tuple(p[j])])

# make it a column of a pandas dataframe
vectors_df = DataFrame({'all_vecs': all_vecs, 'coordinates': coordinates})

# Here we normalize all those vectors to prepare to take the dot product with the vector called "length vector"
# Dot product will be used to determine orthogonality
vectors_df['all_vecs_normalized'] = vectors_df.all_vecs.apply(lambda x: x / norm(x))

# Take the dot product
#vectors_df['dot_product'] = vectors_df.all_vecs_normalized.apply(lambda x: np.dot(x, length_axis))
vectors_df['dot_product'] = vectors_df.all_vecs_normalized.apply(lambda x: np.dot(x, unit_length_vector))
#vectors_df['orthogonal'] = vectors_df.dot_product.apply(lambda x: x < 0.15)

vectors_df['norms'] = vectors_df.all_vecs.apply(lambda x: norm(x))

width = nanmax(vectors_df[vectors_df.dot_product < 0.15].norms)
width_coords = vectors_df[vectors_df.norms == width].coordinates.tolist()[0]
pixelwidth = round(width, 2)















