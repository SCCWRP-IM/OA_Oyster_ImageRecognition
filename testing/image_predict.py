#imagepredict
# Source: https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Prediction/README.md

from imageai.Prediction import ImagePrediction
import os 

execution_path = "P:/PartTimers/ZaibQuraishi/ImageCountProject"
prediction = ImagePrediction()
'''
    Using ResNet50 (seems like the best option)     !!!!!!!!!!!!!!
Observations:
    (1) petri dish is recogized (when result_count >= 4)
    (2) ruler MAY be recognized as "rule" (when result_count >= 1)
    (3) no regonition of oyster
    (4) MANY different objects are recognized
'''
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))

'''
    Using SqueezeNet --- let's NOT use this one
Observations:
    (1) petri dish recognized (when result_count >= 11)
    (2) ruler is not recognized
    (3) no recogntion of oyster
    (4) MANY different objects are recognized
'''
#prediction.setModelTypeAsSqueezeNet()
#prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
'''
    Using Inception
Observations: 
    (1) petri dish recognized (when result_count >= 1)
    (2) ruler is not recognized
    (3) no recognition of oyster
    (4) MANY different objects are not recognized
'''
#prediction.setModelTypeAsInceptionV3()
#prediction.setModelPath(os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
'''
    Using DenseNet
Observations: 
    (1) petri dish recognized (when result_count >= 3)
    (2) ruler is recognized as "rule" (when result_count >= 2), "slide_rule" (when results_count >= 4), "scale" (when results_count >= 7)
    (3) no recognition of oyster
    (4) MANY different objects are recognized
'''
#prediction.setModelTypeAsDenseNet()
#prediction.setModelPath(os.path.join(execution_path, "DenseNet-BC-121-32.h5"))
'''
    Loading the Model
'''
prediction.loadModel()

#default for result_count = 2
predictions, probabilities = prediction.predictImage(
        os.path.join(execution_path, "photos/oysterref_orig.JPG"), result_count = 20)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)