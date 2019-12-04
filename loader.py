from numpy import empty
import pickle
from dataset import Dataset
from image import Image

f = open("./facedata/facedatatrain", "r")
reader = f.readlines()

labels = open("./facedata/facedatatrainlabels", "r")
lreader = labels.readlines()

trainImageList = []

j = 0
i = 0
while(j < len(reader)):
    image_array = empty([70, 60])
    for r in range(0, 70):
        row = reader[j]
        j += 1
        for c in range(0, 60):
            if row[c] == '#' or row[c] == '+':
                image_array[r][c] = 1
            else:
                image_array[r][c] = 0
    label = lreader[i]
    i += 1
    image = Image(image_array, label)
    trainImageList.append(image)


f = open("./facedata/facedatatest", "r")
reader = f.readlines()

labels = open("./facedata/facedatatestlabels", "r")
lreader = labels.readlines()

testImageList = []

j = 0
i = 0
while(j < len(reader)):
    image_array = empty([70, 60])
    for r in range(0, 70):
        row = reader[j]
        j += 1
        for c in range(0, 60):
            if row[c] == '#' or row[c] == '+':
                image_array[r][c] = 1
            else:
                image_array[r][c] = 0
    label = lreader[i]
    i += 1
    image = Image(image_array, label)
    testImageList.append(image)

f = open("./facedata/facedatavalidation", "r")
reader = f.readlines()

labels = open("./facedata/facedatavalidationlabels", "r")
lreader = labels.readlines()

valImageList = []

j = 0
i = 0
while(j < len(reader)):
    image_array = empty([70, 60])
    for r in range(0, 70):
        row = reader[j]
        j += 1
        for c in range(0, 60):
            if row[c] == '#' or row[c] == '+':
                image_array[r][c] = 1
            else:
                image_array[r][c] = 0
    label = lreader[i]
    i += 1
    image = Image(image_array, label)
    valImageList.append(image)

dataset = Dataset(trainImageList, testImageList, valImageList)
output_file = open('faces_dataset', 'wb')
pickle.dump(dataset, output_file)

