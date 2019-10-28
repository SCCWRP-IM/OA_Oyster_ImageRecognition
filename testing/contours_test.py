import cv2
import numpy as np

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

img = cv2.imread("P:/PartTimers/ZaibQuraishi/ImageCountProject/photos/oyster_test.JPG")
img = image_resize(img, height = 800)
shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
ret,thresh = cv2.threshold(gray,127,255,0)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img,contours,-1,(0,255,0),3)

for h,cnt in enumerate(contours):
    mask = np.zeros(gray.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    mean = cv2.mean(img,mask = mask)

''''
'''

import numpy as np
import cv2
 
im = cv2.imread('photos/oyster_test.JPG')
im = image_resize(im, height = 800)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im,contours[169],-1,(0,255,0),3)


for h,cnt in enumerate(contours):
    if len(cnt) < 30:
        continue
    for point in cnt:
        x = point[0][0]
        y = point[0][1]
    mask = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    mean = cv2.mean(im,mask = mask)
    cv2.imshow("draw indiv contours", mean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




'''
for h,cnt in enumerate(contours):
    print(cv2.moments(cnt))
'''

cv2.imshow("draw indiv contours", im)
cv2.waitKey(0)
cv2.destroyAllWindows()


