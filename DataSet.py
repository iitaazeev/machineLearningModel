import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random
import pickle

# DATADIR = "D:/Jorda/Pictures/ml/hand_gestures_-_Google_Search/" custom search google shit
# DATADIR = "C:/Users/Jorda/Desktop/hand-gestures-recognition/train" sign language

# ^^^^ these are old datasets, which didnt really work with my model ^^^^ 
DATADIR = "D:/Jorda/Pictures/Camera Roll"

# CATEGORIES = ["ok", "thumbs_up"]  # this is the name of the files in which the pictures are in
CATEGORIES = ["ok", "thumb"]
for category in CATEGORIES:  # ok and the thumbs up folders
    path = os.path.join(DATADIR, category)  # create the path to the directory containing categories
    for img in os.listdir(path):  # iterate over each image per thumbs up and ok gesture
        img_array = cv2.imread(os.path.join(path, img),
                               cv2.IMREAD_GRAYSCALE)  # convert to array. we also convert to gray scale because rgb
        # is 3x the size of gray scale
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  # ...and one more!

    print(img_array)

    print(img_array.shape)

IMG_SIZE = 80

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

training_data = []

def create_training_data():
    for category in CATEGORIES:  # for two gestures

        path = os.path.join(DATADIR,category)  # create path to two diff gesture files
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=ok 1=thumb

        for img in tqdm(os.listdir(path)):  # iterate over each image per image of ok and thumbs up
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

            #^^^^^^ this is just an error prevention, probably can ignore it)^^^^^^^^^

create_training_data()

print(len(training_data))

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []  # capital x is your feature set
y = []  # lowercase y is your labels

for features, label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)                  #-1 as a catch all features, img_size x img_size, and a 1 because it is a grey scale
y = np.array(y)

# this is just to save your data... numpy.save is another way it doesnt really matter as long as you save it

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
