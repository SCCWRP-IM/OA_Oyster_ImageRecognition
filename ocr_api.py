#####################################################################################################
#                                                                                                   #
#   This code is from Microsoft's tutorial on how to use their computervision API                   #
#   docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/python-sdk    #
#                                                                                                   #
#####################################################################################################

# ENDPOINTS
endpoint1 = "https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/"
endpoint2 = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"
endpoint3 = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.1/"

# KEYS
key1 = "acc7313e80374a31a860bd8c9de76953"
key2 = "b7c16e1975ab488899414161185d0f82"

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

import os
import sys
import time

'''
os.system('export COMPUTER_VISION_SUBSCRIPTION_KEY="%s"' % endpoint1)
os.system('export COMPUTER_VISION_ENDPOINT="%s"' % key1)

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
'''

# AHHHHHH
#computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
computervision_client = ComputerVisionClient(endpoint1, CognitiveServicesCredentials(key1))

remote_image_url = "https://raw.githubusercontent.com/SCCWRP-IM/OA_Oyster_ImageRecognition/master/OysterShape.jpg"
#remote_image_url = "http://data.sccwrp.org/tmp/IMG_9823.JPG"
local_image_path = "photos/IMG_9823.jpg"
local_image_path = open(local_image_path)
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

