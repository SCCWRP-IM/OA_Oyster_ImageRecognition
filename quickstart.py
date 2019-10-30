from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

import os
import sys
import time
import cv2 as cv



#print(os.environ)

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

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

remote_image_url = "http://data.sccwrp.org/tmp/oysters/2019_08_09_0150.JPG"


# Call API with URL and raw response (allows you to get the operation location)
recognize_printed_results = computervision_client.batch_read_file(remote_image_url,  raw=True)


# Get the operation location (URL with an ID at the end) from the response
operation_location_remote = recognize_printed_results.headers["Operation-Location"]
print(operation_location_remote)
# Grab the ID from the URL
operation_id = operation_location_remote.split("/")[-1]
print(operation_id)

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
            text_results[line.text] = line.bounding_box
            print()

image = cv.imread("/unraid/photos/OAImageRecognition/2019_08_09_0150.JPG")
print(image)
for key in text_results.keys(): # recall that the value associated with each key is a list
    # coords = coordinates. each list in the dictionary are the sets of pixel points for the bounding box of recognized text
    coords = text_results[key]
    min_x = int(min([x for x in coords if coords.index(x) % 2 == 0]))
    min_y = int(min([y for y in coords if coords.index(y) % 2 == 1]))
    max_x = int(max([x for x in coords if coords.index(x) % 2 == 0]))
    max_y = int(max([y for y in coords if coords.index(y) % 2 == 1]))
    cv.rectangle(image, tuple([min_x, min_y]), tuple([max_x, max_y]), (0,0,255))
    print(key)
    del coords


cv.imwrite("cropped_photos/2019_08_09_0150-boxed.jpg", image)


