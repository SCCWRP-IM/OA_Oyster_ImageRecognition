import glob
import os

images_full_paths = glob.glob("/unraid/photos/OAImageRecognition/*.JPG")
image_ids = [x.split("/")[-1].split(".")[0] for x in images_full_paths]

for image_id in image_ids:
    os.system("python3 measure.py -i %s" % image_id)
