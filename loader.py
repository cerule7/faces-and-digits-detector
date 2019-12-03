from numpy import empty
import pickle
from dataset import Dataset
from image import Image

f = open("./facedata/facedatatrain", "r")
reader = f.readlines()

labels = open("./facedata/facedatatrainlabels", "r")
lreader = labels.readlines()

trainImageList = []

for i in range(0, 451):
    image_array = empty([74, 60])
    for r in range(0, 74):
        row = reader[r]
        for c in range(0, 60):
            if row[c] == '#' or row[c] == '+':
                image_array[r][c] = 1
            else:
                image_array[r][c] = 0
    label = lreader[i]
    image = Image(image_array, label)
    trainImageList.append(image)

f = open("./facedata/facedatatest", "r")
reader = f.readlines()

labels = open("./facedata/facedatatestlabels", "r")
lreader = labels.readlines()

testImageList = []

for i in range(0, 150):
    image_array = empty([74, 60])
    for r in range(0, 74):
        row = reader[r]
        for c in range(0, 60):
            if row[c] == '#' or row[c] == '+':
                image_array[r][c] = 1
            else:
                image_array[r][c] = 0
    label = lreader[i]
    image = Image(image_array, label)
    testImageList.append(image)

f = open("./facedata/facedatavalidation", "r")
reader = f.readlines()

labels = open("./facedata/facedatavalidationlabels", "r")
lreader = labels.readlines()

valImageList = []

for i in range(0, 301):
    image_array = empty([74, 60])
    for r in range(0, 74):
        row = reader[r]
        for c in range(0, 60):
            if row[c] == '#' or row[c] == '+':
                image_array[r][c] = 1
            else:
                image_array[r][c] = 0
    label = lreader[i]
    image = Image(image_array, label)
    valImageList.append(image)

dataset = Dataset(trainImageList, testImageList, valImageList)
output_file = open('faces_dataset', 'wb')
pickle.dump(dataset, output_file)

