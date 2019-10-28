import numpy as np
import cv2 as cv
#from scipy.spatial import ConvexHull, convex_hull_plot_2d

from numpy import random, nanmax, nanmin, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt


import imageai
from imageai.Detection.Custom import CustomObjectDetection

import time

import os

from image_resize import image_resize


TIMESTAMP = time.time() * 1000
print(TIMESTAMP)

class Contour:
    def __init__(self, points, original_image):
        "Original image should be the numpy array version, not the path to the image"
        self.points = points
        self.original_image = original_image
        self.max_x = nanmax([x[0] for x in self.points])
        self.max_y = nanmax([y[1] for y in self.points])
        self.min_x = nanmin([x[0] for x in self.points])
        self.min_y = nanmin([y[1] for y in self.points])
        self.window = self.original_image[self.min_x - cushion:self.max_x + cushion, self.min_y - cushion:self.max_y + cushion]
    def crop_window(cushion=0, path)
        cv.imwrite(path, self.window)
        return None
    def containsOysters(self, path, detector):
        try:
            detected_objects = detector.detectObjectsFromImage(
                        input_image=path,
                        output_image_path="output/"+str(TIMESTAMP)+"-contour"+str(i)+"-detected.jpg",
                        minimum_percentage_probability = 10
                    )
            
            oysters = []
            for detection in detected_objects:
                if detection['name'] == 'oyster':
                    oysters.append(detection)
            self.oysters = oysters

            if oysters == []:
                print("unable to detect an oyster in coordinates (%s,%s) to (%s,%s) in contour %s" % (min_x, min_y, max_x, max_y, i))
                print("p[I_row]: %s, p[I_col]: %s" % (p[I_row], p[I_col]))
                self.containsOysters = False
                return False
            else:
                self.containsOysters = True
                return True
        except IOError:
            print("Unable to read in image %s. Try calling the .crop_window() function first and check the file path to ensure it is correct")
            return None

    def matchOysterShape(self, contour_to_match, max_score = 0.3):
        match = cv.matchShapes(contour_to_match, self.points, 1, 0.0)
        print("match score: %s" % match)
        if match > max_score:
            print("the shape of this contour does not sufficiently match the shape of an oyster")
            return False
        else:
            return True
    def getSize(self, theta=range(0,91)):
        size = [[] for lw in range(0,91)] #init size for lw vectors at each theta
        sz = None # initialize the sz variable otherwise it will explode on the first iteration
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
            lengths = [[] for l in range(len(p))] # init lengths list
            widths = [[] for l in range(len(p))] # init widths list
            for l_row in range(0, len(p)):
                lengths[l_row] = dists[l_row][0] # grab all "lengths"
            for w_row in range(0, len(p)):
                widths[w_row] = dists[w_row][1] # grab all "widths"
            l = max(lengths) - min(lengths) # length distance between pts
            w = max(widths) - min(widths) # width distance between pts
            size[angle] = [l, w] # l w vector for each theta
            if sz != np.amax(size, axis=0): # Satisfaction of this condition implies that the value of sz has changed
                sz = np.amax(size, axis=0) # maximum length width for size of object
                max_length_index = lengths.index(max(lengths)) # in the array p, this is the index where we find one of the points along the contour that defines the length
                min_length_index = lengths.index(min(lengths)) # in the array p, this is the index where we find one of the points along the contour that defines the length
                max_width_index = widths.index(max(widths)) # in the array p, this is the index where we find one of the points along the contour that defines the width
                min_width_index = widths.index(min(widths)) # in the array p, this is the index where we find one of the points along the contour that defines the width
                max_length_coord = tuple(p[max_length_index]) # These are for the sake of drawing the lines later
                min_length_coord = tuple(p[min_length_index])
                max_width_coord = tuple(p[max_width_index])
                min_width_coord = tuple(p[min_width_index])
                length = nanmax(sz[0], sz[1])
                width = nanmin(sz[0], sz[1])
        self.length = length
        self.width = width
        self.max_length_coord = max_length_coord
        self.min_length_coord = min_length_coord
        self.max_width_coord = max_width_coord
        self.min_width_coord = min_width_coord
        return {'length':length,'width',width}
        
    def drawLengthAndWidth(self):
        cv.line(self.original_image, self.min_length_coord, self.max_length_coord, (0,255,0))
        cv.line(self.original_image, self.min_width_coord, self.max_width_coord, (0,255,0))
        cv.putText(self.original_image, "L:%spx, W:%spx", self.max_length_coord, FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        return None
        
        



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
    

    return {'length': length, 'width': width, 'sizes': size, 'size': sz, 'max_length_coord': max_length_coord, 'min_length_coord':min_length_coord, 'max_width_coord': max_width_coord, 'min_width_coord': min_width_coord}



# Have to comment out the imshows and stuff like that because I'm running this stuff in a docker container
# it will break as soon as it realizes it can't access the monitors

im = cv.imread('photos/2019_08_08_0060.jpg')
im = image_resize(im, height = 800)


imgray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
imgray = cv.GaussianBlur(imgray, (3,3), 0)
#thresh is bad
#230 works for oysterref.jpg # assumption: this is the quality of the image we will get
#100 for oyster_test.jpg #bas image quality
ret,thresh = cv.threshold(imgray, 185, 255, cv.THRESH_BINARY_INV) # change 1st number fr shadows of shapes
# canny is irrelevant here so let's ignore it despite its use in size_test.py
# imcanny = cv.Canny(imgray, 50, 100)
contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# the oysters are at the following indicies: 169, 189 For the photo oyster_test.jpg



# Here we get a generic oyster shape to match with the detected contours from the input image
oyster_outline = cv.imread("photos/OysterShape.jpg")
oyster_outline = image_resize(oyster_outline, height = 800)
oyster_outline = cv.cvtColor(oyster_outline, cv.COLOR_BGR2GRAY)
_, oyster_outline = cv.threshold(oyster_outline, 127, 255, cv.THRESH_BINARY_INV) # threshold returns a tuple...

# Here we will get a standard contour to compare the others with
contour_standard, h = cv.findContours(oyster_outline, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

# contour standard is actually a list of al detected contours from that image, 
#   and the largest one is actually the outline of the entire image
sorted_contour_standard = sorted(contour_standard, key = cv.contourArea, reverse = True)

# There are only two items in the list, grab the second item and that it going to be the one that we want
contour_standard = sorted_contour_standard[1]

# save memory
del sorted_contour_standard

# prepare the object detector
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("object_detection/models/detection_model-ex-001--loss-0054.676.h5")
detector.setJsonPath("object_detection/json/detection_config.json")
detector.loadModel()

for i in range(len(contours)):
    # here we grab the contour
    cv.drawContours(im,contours[i],-1,(0,255,0),3)

    # contours are just lists of lists of coordinate points 
    # (on a basic simple level. I think they are technically np arrays or something)
    # Anyways, here we are extracting the list of coordinate points that define each contour to then analyze it
    points = np.ndarray.tolist(contours[i])

    #change range arg
    p = [[] for i in range(len(contours[i]))]

# point is a list of lists of lists.
    # we need only a list of lists
    for j, coord in enumerate(points):
        #print(j, coord)
        p[j] = points[j][0]    
    del points
    
    contour = Contour(p, im)
    cropped_path = "cropped_photos/contour-%s.jpg" % i
    contour.crop_window(cushion=15, path = cropped_path)
    if contour.containsOysters(path=cropped_path, detector=detector) and contour.matchOysterShape(contour_standard):
        contour.getSize()
        print("contour %s represents and oyster of length %s and width %s" % (i, contour.length, contour.width))
        contour.drawLengthAndWidth()
    else:
        continue



    


p = measure(image=im,contours=contours,contour_standard=contour_standard,detector=detector)
cv.imwrite("output_images/"+str(TIMESTAMP)+"-finaloutput.jpg", im)

#######################################33
# CODE ARCHIVE
#################
'''
    # here we grab the contour
    cv.drawContours(im,contours[i],-1,(0,255,0),3)


    # contours are just lists of lists of coordinate points 
    # (on a basic simple level. I think they are technically np arrays or something)
    # Anyways, here we are extracting the list of coordinate points that define each contour to then analyze it
    points = np.ndarray.tolist(contours[i])

    #change range arg
    p = [[] for i in range(len(contours[i]))]

    for j, coord in enumerate(points):
        #print(j, coord)
        p[j] = points[j][0]
    
    del points
    
    # we want to loop through the points contained in p for each contour. 
    # p is a set of points that form the contour so the contours contains a set of p per contour drawn
    # compute distance
    D = pdist(p)
    # input into an array
    D = squareform(D);
    # find max distance and which points this corresponds to
    N = nanmax(D)
    [I_row, I_col] = unravel_index( argmax(D), D.shape )

    # now, it appears that p[I_col] represents the point (x,y) on the left of the contour, and p[I_col] is the one on the right
    # The two points are the ones that are farthest away from each other in the contour
    

    # Now we can find the maximum orthogonal distance to the length along the contour, in other words, the width
    length_vector = np.array(p[I_col]) - np.array(p[I_row])
    
    if contour_info['length'] < 50:
        # If the contour is too small, there is no way it represents an oyster
        # We will not apply the same reasoning to large contours, since multiple oysters may be 
        #   considered as one contour, in which case we have to determine why the contour is that large
        #   It may be due to multiple oysters being lumped into one
        #   It may also be an oyster combined with another unwanted object in the photo
        #   It may also very well be something completely irrelevant
        del N
        del D
        del p
        continue
    

    # Here we will get a box that encloses the oyster contour, and then run it through the recognition model
    min_x = nanmin([x[0] for x in p])
    min_y = nanmin([y[1] for y in p])

    max_x = nanmax([x[0] for x in p])
    max_y = nanmax([y[1] for y in p])
     
    min_x = nanmax([min_x - 25, 0])
    min_y = nanmax([min_y - 25, 0])

    max_x = nanmin([max_x + 25, im.shape[1]]) # im.shape returns a tuple (height, width, channels)
    max_y = nanmin([max_y + 25, im.shape[0]])

    # Here we subset the image to the smallest box that contains the entire contour in question, 
    #   so we can run it through the oyster detector and see if this thing has an oyster in it or not
    cropped = im[min_y:max_y, min_x:max_x]

    # So we can trace the contour back, we write it to a filename that has the timestamp along with the contour number
    cropped_path = "cropped_photos/"+str(TIMESTAMP)+"-contour"+str(i)+".jpg"
    cv.imwrite(cropped_path, cropped)
    del cropped

    # Next step is to see if we have an oyster in this thing or not.
    # If yes, then we'll go ahead and measure it.
    # if not, we'll skip this contour and not execute the remaining code.
    # If the detector detects an oyster, and the max length across the contour exceeds the max_length argument
    #   then we will go ahead and make a note of it, saying that this contour contains an oyster, but the measurement will
    #   not likely be accurate since the contour most likely includes an unwanted object

    # Start oyster detection here
    detections = detector.detectObjectsFromImage(
                input_image=cropped_path,
                output_image_path="output/"+str(TIMESTAMP)+"-contour"+str(i)+"-detected.jpg",
                minimum_percentage_probability = 10
            )
    
    if detections == []:
        print("unable to detect an oyster in coordinates (%s,%s) to (%s,%s) in contour %s" % (min_x, min_y, max_x, max_y, i))
        del detections
        print("p[I_row]: %s, p[I_col]: %s" % (p[I_row], p[I_col]))
        continue

    # Here we filter based on how much the contour shape matches the general shape of an oyster
    match = cv.matchShapes(contour_standard, cont, 1, 0.0)
    print("match score for contour %s: %s" % (i, match))
    
    if match > 0.3:
        print("the shape of contour %s does not sufficiently match the shape of an oyster" % i)
        del match
        #os.system("rm %s" % cropped_path)
        print("p[I_row]: %s, p[I_col]: %s" % (p[I_row], p[I_col]))
        continue
    del match

    #print("detections")
    #print(detections)
    #print("type(detections)")
    #print(type(detections))
    print("here are each of the individual detections for contour %s" % i)
    for detection in detections:
        print(type(detection))
        print(detection)

    # Based on the detections 


    #draw line in original image (we will use this for the bounding boxes)
    cv.line(im, tuple(p[I_row]), tuple(p[I_col]), (0, 0, 255), 2)
    
    print("p[I_row]: %s, p[I_col]: %s" % (p[I_row], p[I_col]))
    
    
'''
'''
cv.imshow("draw indiv contours", im)
cv.waitKey(0)
cv.destroyAllWindows()
'''


'''
cv.imshow("threshed", thresh)
cv.waitKey(0)
cv.destroyAllWindows()
'''
