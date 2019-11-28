import pickle
import numpy

def featureVector(image, num_features):
    quadrants = numpy.split(image.image, num_features)
    featureVector = []
    for q in quadrants:
        print(q)
        featureVector.append(numpy.count_nonzero(q == 1))
    return featureVector

digits = pickle.load(open( "digits_dataset", "rb" ) )
faces = pickle.load(open( "face_dataset", "rb" ) )

print(featureVector(digits.trainData[0], 28))
print(featureVector(faces.trainData[0], 74))
