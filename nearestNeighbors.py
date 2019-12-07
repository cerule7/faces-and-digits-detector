from utils import featureVector

def vectorDifference(vector1, vector2):
    difValue = 0
    for i in vector1:
        difValue += abs(vector1[i] - vector2[i])
    return difValue

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
    return modeOfKNeighbors(trainData, differenceValues, 20)
    #return trainData[differenceValues.index(min(differenceValues))].label

     #the min of differenceValues will be the closest neighbor in the test data
     #to the image, the index of that min in differenceValues aligns with the
     #index of corresponding image in the train set and thus corresponds to that image’s label index as well.
     #Thus, this function when used with face data will return 0 or 1 and when used with digit data will return an int from 0 to 9
