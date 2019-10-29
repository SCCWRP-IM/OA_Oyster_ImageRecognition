#####################################################################################################
#                                                                                                   #
#   This code is from Microsoft's tutorial on how to use their computervision API                   #
#   docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/python-sdk    #
#                                                                                                   #
#####################################################################################################
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TestRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentialsd

import os
import sys
import time


if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("Please set the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable")
    sys.exit()

if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
else:
    print("Please set the COMPUTER_VISION_ENDPOINT environment variable")
    sys.exit()

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

remote_image_url = "http://data.sccwrp.org/tmp/oysters/IMG_9823.JPG"
recognize_printed_results = computervision_client.batch_read_file(remote_image_url, raw = True)

while True:
    get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
    if get_printed_text_results.status not in ['NotStarted','Running']:
        break
    time.sleep(1)

if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
    for text_result in get_printed_text_results.recognition_results:
        for line in text_result.lines:
            print(line.text)
            print(line.bounding_box)
print()

