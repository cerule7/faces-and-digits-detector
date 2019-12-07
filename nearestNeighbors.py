from utils import featureVector
from scipy import spatial

def vectorDifference(vector1, vector2):
    return spatial.distance.cosine(vector1, vector2)

def modeOfKNeighbors(trainData, distances, k):
    distances.sort(key = lambda x: x[0]) 
    neighbors = distances[0:k:]
    labels = [int(trainData[index[1]].label) for index in neighbors]
    return mode(labels)

def mode(labels):
    counter = {}
    for label in labels:
        if label in counter:
            counter[label] += 1
        else:
            counter[label] = 1
    return max(counter, key=lambda key: counter[key])

def NNClassifier(image, trainData, trainFeatureVectors, r, c, A, Y):
    imageFeatures = featureVector(image.image, r, c, A, Y)
    differenceValues = [(vectorDifference(imageFeatures, trainFeatureVectors[i]), i) for i in range(0, len(trainFeatureVectors))]  #vector of how the images features compare to training images features
    return modeOfKNeighbors(trainData, differenceValues, 7)