from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("Cropped/models/model_ex-020_acc-0.935115.h5")
prediction.setJsonPath("Cropped/json/model_class.json")
prediction.loadModel(num_objects=3)

#predictions, probabilities = prediction.predictImage("Cropped/train/vineyard/tiles-62.png", result_count=3)
predictions, probabilities = prediction.predictImage("test.jpg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)