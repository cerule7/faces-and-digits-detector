from numpy import empty
import pickle
from dataset import Dataset
from image import Image

f = open("./digitdata/trainingimages", "r")
reader = f.readlines()

labels = open("./digitdata/traininglabels", "r")
lreader = labels.readlines()

trainImageList = []

j = 0
i = 0
while(j < len(reader)):
    image_array = empty([28, 28])
    for r in range(0, 28):
        row = reader[j]
        j += 1
        for c in range(0, 28):
            if row[c] == '#' or row[c] == '+':
                image_array[r][c] = 1
            else:
                image_array[r][c] = 0
    label = lreader[i]
    i += 1
    image = Image(image_array, label)
    trainImageList.append(image)


f = open("./digitdata/testimages", "r")
reader = f.readlines()

labels = open("./digitdata/testlabels", "r")
lreader = labels.readlines()

testImageList = []

j = 0
i = 0
while(j < len(reader)):
    image_array = empty([28, 28])
    for r in range(0, 28):
        row = reader[j]
        j += 1
        for c in range(0, 28):
            if row[c] == '#' or row[c] == '+':
                image_array[r][c] = 1
            else:
                image_array[r][c] = 0
    label = lreader[i]
    i += 1
    image = Image(image_array, label)
    testImageList.append(image)

f = open("./digitdata/validationimages", "r")
reader = f.readlines()

labels = open("./digitdata/validationlabels", "r")
lreader = labels.readlines()

valImageList = []

j = 0
i = 0
while(j < len(reader)):
    image_array = empty([28, 28])
    for r in range(0, 28):
        row = reader[j]
        j += 1
        for c in range(0, 28):
            if row[c] == '#' or row[c] == '+':
                image_array[r][c] = 1
            else:
                image_array[r][c] = 0
    label = lreader[i]
    i += 1
    image = Image(image_array, label)
    valImageList.append(image)

dataset = Dataset(trainImageList, testImageList, valImageList)
output_file = open('digits_dataset', 'wb')
pickle.dump(dataset, output_file)

