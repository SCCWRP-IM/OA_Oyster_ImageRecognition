import numpy as np
from scipy.stats import linregress
import cv2
#finding the width
#this uses information from the testing_distance.py script

test = [p[I_row], p[I_col]]
test_linreg = linregress(test[0], test[1])
diff = np.array(test[1]) - np.array(test[0])
midpt = diff/2

slope = test_linreg[0]
intercept = test_linreg[1]

pslope = 1/(-slope)
b = pslope*p[0][0] - p[0][1]

for i,pt in enumerate(p):
    b[i] = pslope*pt[0] - pt[1]
    


### using information from testing_distance.py
# here, we are trying to create a mask to pick up on only the objects we want
# and determine a strong threshold value
plt.plot(histogram) # <- or here
plt.show()
histogram = cv2.calcHist(images = [imgray], 
    channels = [0], 
    mask = None, 
    histSize = [256], 
    ranges = [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0, 256]) # <- named arguments do not work here

plt.plot(histogram) # <- or here
plt.show()
channels = cv2.split(imgray)
colors = ("b", "g", "r") 

# create the histogram plot, with three lines, one for
# each color 
plt.xlim([0, 256])
for(channel, c) in zip(channels, colors):
    histogram = cv2.calcHist(
        images = [channel], 
        channels = [0], 
        mask = None, 
        histSize = [256], 
        ranges = [0, 256])

    plt.plot(histogram, color = c)

plt.xlabel("Color value")
plt.ylabel("Pixels")

plt.show()