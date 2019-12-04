import pickle

# digit: r = 14, c = 14, A = 28, Y = 28
# face: r = 10, c = 10, A = 70, Y = 60
def featureVector(image, r, c, A, Y):
    featVector = list() #final feature vector to be returned as output
    featWidth = int(A / r)  #how many pixels make the horizontal dimension of a feature rectangle
    featHeight = int(Y / c) ##how many pixels make the vertical dimension of a feature rectangle
    for i in range(0, r):
        maxHorizontal = i*featWidth + featWidth
        verticalPosition = 0
        featCount = 0 #counts number of black pixels in feature rectangle
        for j in range(0, Y):
            if(verticalPosition % featHeight == 0):
                featVector.append(featCount)
                featCount = 0
            verticalPosition += 1
            for k in range(i * featWidth, maxHorizontal):
                if (image[k][j] == 1.0):
                    featCount += 1
    return featVector

# faces = pickle.load(open( "faces_dataset", "rb"))
# print(featureVector(faces.testData[0].image, 10, 10, 70, 60))
# print(featureVector(faces.testData[5].image, 10, 10, 70, 60))
# print(faces.testData[0].image == faces.testData[1].image)
# print(featureVector(faces.testData[1].image, 10, 10, 70, 60) == featureVector(faces.testData[0].image, 10, 10, 70, 60))