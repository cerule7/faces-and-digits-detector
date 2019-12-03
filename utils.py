import numpy

# featureVector(digits.trainData[0], 28)
# featureVector(faces.trainData[0], 74)
def featureVector(image, num_features):
    quadrants = numpy.split(image.image, num_features)
    featureVector = []
    for q in quadrants:
        featureVector.append(numpy.count_nonzero(q == 1))
    return featureVector
