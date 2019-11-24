from numpy import empty
import pickle

f = open("./facedata/facedatatrain", "r")
reader = f.readlines()

labels = open("./facedata/facedatatrainlabels", "r")
lreader = labels.readlines()

imageList = []
labelList = []

for i in range(0, 451):
    image = empty([70, 60])
    for r in range(0, 70):
        row = reader[r]
        for c in range(0, 60):
            if row[c] == '#' or row[c] == '+':
                image[r][c] = 1
            else:
                image[r][c] = 0
    label = lreader[i]
    labelList.append(label)
    imageList.append(image)
    if i % 50 == 0:
        print(label)
        print(image)

output_file = open('processed_train_images', 'wb')
pickle.dump(imageList, output_file)

output_file = open('processed_train_labels', 'wb')
pickle.dump(labelList, output_file)
