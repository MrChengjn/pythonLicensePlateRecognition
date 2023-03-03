#搭建模型
import copy

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
#训练路径设置
class eng:
	def __init__(self):
		self.MODEL_PATH = "model/eng.h5"
		self.IMAGE_WIDTH = 20
		self.IMAGE_HEIGHT = 20
		self.CLASSIFICATION_COUNT = 34
		self.LABEL_DICT = {
			'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
			'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17,	'J':18, 'K':19,
			'L':20, 'M':21, 'N':22, 'P':23, 'Q':24, 'R':25, 'S':26, 'T':27, 'U':28, 'V':29,
			'W':30, 'X':31, 'Y':32, 'Z':33
		}

		self.model=keras.models.load_model(self.MODEL_PATH)


#预测特定图片(test)
	def testbegin(self,img):
		resized_image = cv.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
		resized_image=resized_image.reshape((-1,self.IMAGE_WIDTH, self.IMAGE_HEIGHT,1))
		dicxx=list(np.squeeze(self.model.predict(resized_image)))
		num=dicxx.index(max(dicxx))
		answ=list(self.LABEL_DICT.keys())[list(self.LABEL_DICT.values()).index(num)]
		return answ


"""
def load_data(dir_path):
    data = []
    labels = []

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
                resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                data.append(resized_image)
                labels.append(LABEL_DICT[item])

    return np.array(data), np.array(labels)


def normaldata(data):
    return (data - data.mean()) / data.max()


def hotten(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots
train_data, train_labels = load_data(TRAIN_DIR)
print(train_data.shape)
train_data = normaldata(train_data)
train_labels = hotten(train_labels)
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
#输出层
model.add(Dense(34, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
model.fit(train_data,train_labels,batch_size=50, epochs=40)
model.save(MODEL_PATH)
"""