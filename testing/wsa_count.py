#watershed by open cv
from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import imutils
import cv2

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
    return(resized)

img = cv2.imread("P:/PartTimers/ZaibQuraishi/ImageCountProject/photos/oyster_test.JPG")
img = image_resize(img, height = 800)
shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#shifted to gray
#gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

#image to gray (catches 12, with contours)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#image to gray (only catches 1, with contours)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Here, there are only 12 contours being found. So we need to make sure that the 
correct contours are being found. 
'''

contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
print("[INFO] {} unique contours found".format(len(contours)))

for (i,c) in enumerate(contours):
    ((x,y), _) = cv2.minEnclosingCircle(c)
    cv2.putText(img, "#{}s".format(i + 1), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

cv2.imshow("draw indiv contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


