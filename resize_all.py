import cv2 as cv
import numpy as np
import glob
import os
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
print("getting list of photos to resize")
photos = glob.glob("/unraid/photos/OAImageRecognition/NewPhotos/*.JPG")
print("looping through photos")
for photo in photos:
    print("image_id")
    image_id = str(photo).split("/")[-1].split(".")[0]
    print("read in the image")
    pic = cv.imread(photo)
    print("resize the image")
    pic = image_resize(pic, height = 800)
    print("write image out to a JPG")
    cv.imwrite("/unraid/photos/OAImageRecognition/NewPhotos/resized/%s-resized.JPG" % image_id, pic)


