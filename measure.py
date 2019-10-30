import numpy as np
import cv2 as cv
from pandas import Series, DataFrame, isnull
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from numpy import random, nanmax, nanmin, argmax, unravel_index
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt
import imageai
from imageai.Detection.Custom import CustomObjectDetection
import time
import os
import sys

# Get the pixels per millimeter ratio
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

# Add your Computer Vision subscription key to your environment variables.
if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
# Add your Computer Vision endpoint to your environment variables.
if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
else:
    print("\nSet the COMPUTER_VISION_ENDPOINT environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()

# authenticate the API credentials so microsoft knows we have access
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# URL to the image we are analyzing
remote_image_url = "http://data.sccwrp.org/tmp/oysters/IMG_9823.JPG"

# Call API with URL and raw response (allows you to get the operation location)
recognize_printed_results = computervision_client.batch_read_file(remote_image_url,  raw=True)

# Get the operation location (URL with an ID at the end) from the response
operation_location_remote = recognize_printed_results.headers["Operation-Location"]
print(operation_location_remote)

# Grab the ID from the URL
operation_id = operation_location_remote.split("/")[-1]

# Call the "GET" API and wait for it to retrieve the results 
while True:
    get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
    if get_printed_text_results.status not in ['NotStarted', 'Running']:
        break
    time.sleep(1)

#Print the detected text, line by line
# We will also go ahead and store the results in a python dictionary
text_results = dict()
if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
    for text_result in get_printed_text_results.recognition_results:
        for line in text_result.lines:
            print(line.text)
            print(line.bounding_box)
            text_results[line.text] = [int(x) for x in line.bounding_box]
            print()


oyster_species = None
species_text = None
species_number = None

for key in text_results.keys():
    if 'pacific' in key.lower():
        oyster_species = 'Pacific'
        species_text = key
        species_number = re.split('\s+', species_text)[0]
    elif 'olympiad' in key.lower():
        oyster_species = 'Olympiad'
        species_text = key
        species_number = re.split('\s+', species_text)[0]
    else:
        print("unable to identify oyster species")
    

# On the ruler, the AQUATIC or ECO-SYSTEMS,INC. text appeared to be about 30mm maybe 30.5
# the bounding box typically has a little room around it so I think we can go ahead and say 30.5mm    

if "ECO-SYSTEMS,INC." in text_results.keys():
    # Eco-systems,inc. seemed to be most reliable in terms of the box being tight on the text.
    min_x = min([x for x in text_results['ECO-SYSTEMS,INC.'] if text_results['ECO-SYSTEMS,INC.'].index(x) % 2 == 0])
    max_x = max([x for x in text_results['ECO-SYSTEMS,INC.'] if text_results['ECO-SYSTEMS,INC.'].index(x) % 2 == 0])
    min_y = min([y for y in text_results['ECO-SYSTEMS,INC.'] if text_results['ECO-SYSTEMS,INC.'].index(y) % 2 == 1])
    max_y = max([y for y in text_results['ECO-SYSTEMS,INC.'] if text_results['ECO-SYSTEMS,INC.'].index(y) % 2 == 1])
    pixel_length = max([max_y - min_y, max_x - min_x])   
    mm_pixel_ratio = np.true_divide(30.5, pixel_length)
 
elif set(['AGUNTIC', 'AQUATIC']).intersection(set(text_results.keys())) != set():
    if 'AGUNTIC' in text_results.keys():
        text_results['AQUATIC'] = text_results['AGUNTIC']
        del text_results['AGUNTIC']
    
    min_x = min([x for x in text_results['AQUATIC'] if text_results['AQUATIC'].index(x) % 2 == 0])
    max_x = max([x for x in text_results['AQUATIC'] if text_results['AQUATIC'].index(x) % 2 == 0])
    min_y = min([y for y in text_results['AQUATIC'] if text_results['AQUATIC'].index(y) % 2 == 1])
    max_y = max([y for y in text_results['AQUATIC'] if text_results['AQUATIC'].index(y) % 2 == 1])
    pixel_length = max([max_y - min_y, max_x - min_x])
    mm_pixel_ratio = np.true_divide(30.5, pixel_length)

else:
    # If those aren't recognized, we fall back on text that always has to be in there (if they took the photo correctly
    min_x = min([x for x in text_results[species_text] if text_results[species_text].index(x) % 2 == 0])
    max_x = max([x for x in text_results[species_text] if text_results[species_text].index(x) % 2 == 0])
    min_y = min([y for y in text_results[species_text] if text_results[species_text].index(y) % 2 == 1])
    max_y = max([y for y in text_results[species_text] if text_results[species_text].index(y) % 2 == 1])
    pixel_length = max([max_y - min_y, max_x - min_x])   

# First need to talk to Darrin about how I can physically measure these things.
# Didn't find them in the Exposure lab.
''' 
    if oyster_species = 'Pacific':
        if len(species_number == 1):
            mm_pixel_ratio = 
        elif len(species_number == 2):
            mm_pixel_ratio = 
        else:
            print("unable to get millimeter to pixel ratio")
    elif oyster_species = 'Olympiad':
        if len(species_number == 1):
            mm_pixel_ratio = 
        elif len(species_number == 2):
            mm_pixel_ratio = 
        else:
            print("unable to get millimeter to pixel ratio")
        
    else:
        print("unable to get millimeter to pixel ratio")
'''
                    



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


class Contour:
    def __init__(self, points, original_image):
        "Original image should be the numpy array version, not the path to the image"
        self.points = points
        self.original_image = original_image
    def crop_window(self, path, image, cushion=0):
        self.max_x = nanmax([x[0] + cushion for x in self.points] + [image.shape[1]])
        self.max_y = nanmax([y[1] + cushion for y in self.points] + [image.shape[0]])
        self.min_x = nanmin([x[0] - cushion for x in self.points] + [0])
        self.min_y = nanmin([y[1] - cushion for y in self.points] + [0])
        self.window = image[self.min_x:self.max_x, self.min_y:self.max_y]
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
                return False
            else:
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
    def getLength(self):
        '''
        Using the method of getting the max distance across and a nearly orthogonal vector of max distance to that one
        Hard to explain in words
        '''
        # we want to loop through the points contained in p for each contour. 
        # p is a set of points that form the contour so the contours contains a set of p per contour drawn
        # compute distance
        D = pdist(self.points)
        # input into an array
        D = squareform(D);
        # find max distance and which points this corresponds to
        self.length = round(nanmax(D), 2)
        # called I_row and I_col since it is grabbing the "ij" location in a matrix where the max occurred.
        # the row number where it occurs represents one of the indices in the original self.points array where one of the points on the contour lies
        # the column number would be the point on the opposite side of the contour
        # L_row, and L_col since these indices correspond with coordinate points that give us the length
        [L_row, L_col] = unravel_index( argmax(D), D.shape )
        
        self.min_length_coord = tuple(self.points[L_row])
        self.max_length_coord = tuple(self.points[L_col])
        self.length_coords = [self.min_length_coord, self.max_length_coord]
        self.length_vector = np.array(self.max_length_coord) - np.array(self.min_length_coord)
        self.unit_length_vector = self.length_vector / norm(self.length_vector)      
 
        return self.length
    def getWidth(self):
        # length axis represents a unit vector along the direction where we found the longest distance over the contour
        # length_axis = (np.array(p[L_col]) - np.array(p[L_row])) / norm(np.array(p[L_col]) - np.array(p[L_row]))
        '''above will be replaced wityh self.unit_length_vector'''
        # length_axis = self.unit_length_vector
       
        # all_vecs will be an list of vectors that are all the combinations of vectors that pass over the contour area
        all_vecs = []
        coordinates = []
        for i in range(0, len(self.points) - 1):
            for j in range(i + 1, len(self.points)):
                all_vecs.append(np.array(self.points[i]) - np.array(self.points[j]))
                coordinates.append([tuple(self.points[i]), tuple(self.points[j])])
        
        # make it a column of a pandas dataframe
        vectors_df = DataFrame({'all_vecs': all_vecs, 'coordinates': coordinates})
        
        # Here we normalize all those vectors to prepare to take the dot product with the vector called "length vector"
        # Dot product will be used to determine orthogonality
        vectors_df['all_vecs_normalized'] = vectors_df.all_vecs.apply(lambda x: x / norm(x))

        # Take the dot product
        #vectors_df['dot_product'] = vectors_df.all_vecs_normalized.apply(lambda x: np.dot(x, length_axis))
        vectors_df['dot_product'] = vectors_df.all_vecs_normalized.apply(lambda x: np.dot(x, self.unit_length_vector))
        #vectors_df['orthogonal'] = vectors_df.dot_product.apply(lambda x: x < 0.15)
       
        vectors_df['norms'] = vectors_df.all_vecs.apply(lambda x: norm(x))

        if any(vectors_df.dot_product < 0.15):
            # allowing dot product to be up to 0.15 allows the length and width to have an angle of 81.37 to 90 degrees between each other
            width = nanmax(vectors_df[vectors_df.dot_product < 0.15].norms)
            self.width_coords = vectors_df[vectors_df.norms == width].coordinates.tolist()[0]
            self.width = round(width, 2)
        else:
            self.width = None
            self.width_coords = None
     

    def drawLengthAndWidth(self, image):
        "image represents the image we are drawing on"
        cv.line(image, self.length_coords[0], self.length_coords[1], (0,255,0))
        cv.line(image, self.width_coords[0], self.width_coords[1], (0,255,0))
        cv.putText(image, "L:%spx, W:%spx" % (self.length, self.width), self.length_coords[1], cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        return None
        

im = cv.imread('/unraid/photos/OAImageRecognition/IMG_9823.jpg')
im = image_resize(im, height = 800)


imgray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
imgray = cv.GaussianBlur(imgray, (3,3), 0)
ret,thresh = cv.threshold(imgray, 185, 255, cv.THRESH_BINARY_INV) # change 1st number fr shadows of shapes
cv.imwrite("output_images/threshed.jpg", thresh)
contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)


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
detector.setModelPath("/home/object_detection/models/detection_model-ex-001--loss-0054.676.h5")
detector.setJsonPath("/home/object_detection/json/detection_config.json")
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
   
    p = np.array(p)

    contour = Contour(p, im)
    cropped_path = "cropped_photos/contour-%s.jpg" % i
    contour.crop_window(path=cropped_path,image=im,cushion=7) # path argument is cropped_path, cushion argument is 15 pixels
    if cv.imread(cropped_path) is not None:
        contour.getLength()
        if contour.length > 40:
            if contour.matchOysterShape(contour_standard) and contour.containsOysters(path=cropped_path, detector=detector):
                contour.getWidth()
                if contour.width is not None:
                    print("contour %s represents and oyster of length %s and width %s" % (i, contour.length, contour.width))
                    contour.drawLengthAndWidth(image=im)
                else:
                    print("unable to get the width of contour %s" % i)
                    continue
            else:
                continue
        else:
            print("skipping contour %s due to unusually short length" % i)
            continue
    else:
        continue

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
   




# From the contour class. This is based on the code from that stack overflow post that Zaib found 
    def getSize_old(self, theta=range(0,91)):
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
            #print("sz: %s" % sz)
            #print("type(sz) == %s" % type(sz))
            #print("np.amax(size, axis=0): %s" % np.amax(size, axis=0))
            #print("type(np.amax(size, axis=0)) == %s" % type(np.amax(size, axis=0)))
            #print("sz != np.amax(size, axis=0): %s" % sz != np.amax(size, axis=0))
            #print(sz != np.amax(size, axis=0))
            if sz != list(np.amax(size, axis=0)): # Satisfaction of this condition implies that the value of sz has changed
                sz = list(np.amax(size, axis=0)) # maximum length width for size of object
                max_length_index = lengths.index(max(lengths)) # in the array p, this is the index where we find one of the points along the contour that defines the length
                min_length_index = lengths.index(min(lengths)) # in the array p, this is the index where we find one of the points along the contour that defines the length
                max_width_index = widths.index(max(widths)) # in the array p, this is the index where we find one of the points along the contour that defines the width
                min_width_index = widths.index(min(widths)) # in the array p, this is the index where we find one of the points along the contour that defines the width
                max_length_coord = tuple(p[max_length_index]) # These are for the sake of drawing the lines later
                min_length_coord = tuple(p[min_length_index])
                max_width_coord = tuple(p[max_width_index])
                min_width_coord = tuple(p[min_width_index])
                length = nanmax([sz[0], sz[1]])
                width = nanmin([sz[0], sz[1]])
        self.length = round(length, 2)
        self.width = round(width, 2)
        self.max_length_coord = max_length_coord
        self.min_length_coord = min_length_coord
        self.max_width_coord = max_width_coord
        self.min_width_coord = min_width_coord
        return {'length':length,'width':width}
    
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
        


'''
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
'''
