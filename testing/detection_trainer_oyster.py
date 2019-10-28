# detection_trainer_oyster
#  https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/Custom/CUSTOMDETECTION.md#objectextraction
# https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/
from imageai.Detection.Custom import DetectionModelTrainer
#import tensorflow as tf

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="P:\PartTimers\ZaibQuraishi\oyster-training")
trainer.setTrainConfig(object_names_array = ["oyster"], batch_size = 4, num_experiments = 100, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()

trainer.evaluateModel(model_path="")
