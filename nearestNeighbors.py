def vectorDifference(vector1, vector2):
    difValue = 0
    for i in vector1:
        difValue += abs(vector1[i]-vector2[i]) #v1 and v2 should be same size since same number of features

    return difValue

def NNClassifier(image, train, labelVector, totalNumFeatures):
    trainFeatureVectors = []
    for i in train:
        trainFeatureVectors.append(featureVector(train[i], totalNumFeatures))
    imageFeatures = featureVector(image, totalNumFeatures)
    differenceValues = []  #vector of how the images features compare to training images features
    for i in trainFeatureVectors:
        differenceValues.append(vectorDifference(imageFeatures, trainFeatureVectors[i]))
    return labelVector[differenceValues.index(min(differenceValues))]
     #the min of differenceValues will be the closest neighbor in the test date
     #to the image, the index of that min in differenceValues aligns with the
     #index of corresponding image in the train set and thus corresponds to that image’s label index as well.
     #Thus, this function when used with face data will return 0 or 1 and when used with digit data will return an int from 0 to 9
