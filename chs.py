# 搭建模型
import tensorflow.keras as keras
import copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt


# 训练路径设置
class chs:
    def __init__(self):
        self.MODEL_PATH = "model/cnn_chs"
        self.model = keras.models.load_model(self.MODEL_PATH)
        self.IMAGE_WIDTH = 24
        self.IMAGE_HEIGHT = 48
        self.CLASSIFICATION_COUNT = 34
        self.LABEL_DICT = {
            'chuan': 0, 'e': 1, 'gan': 2, 'gan1': 3, 'gui': 4, 'gui1': 5, 'hei': 6, 'hu': 7, 'ji': 8, 'jin': 9,
            'jing': 10, 'jl': 11, 'liao': 12, 'lu': 13, 'meng': 14, 'min': 15, 'ning': 16, 'qing': 17, 'qiong': 18,
            'shan': 19,
            'su': 20, 'sx': 21, 'wan': 22, 'xiang': 23, 'xin': 24, 'yu': 25, 'yu1': 26, 'yue': 27, 'yun': 28,
            'zang': 29,
            'zhe': 30
        }
        self.LABEL_DICT_CH = {
            'chuan': '川', 'e': '鄂', 'gan': '赣', 'gan1': '甘', 'gui': '贵', 'gui1': '桂', 'hei': '黑', 'hu': '沪', 'ji': '冀',
            'jin': '津',
            'jing': '京', 'jl': '吉', 'liao': '辽', 'lu': '鲁', 'meng': '蒙', 'min': '闽', 'ning': '宁', 'qing': '青',
            'qiong': '琼', 'shan': 19,
            'su': '苏', 'sx': '晋', 'wan': '皖', 'xiang': '湘', 'xin': '新', 'yu': '豫', 'yu1': '渝', 'yue': '粤', 'yun': '云',
            'zang': '藏',
            'zhe': '浙'
        }
#返回预测结果
    def testbegin(self,img):
        # 预测特定图片(img)s
        resized_image = cv.resize(img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        resized_image = resized_image.reshape((-1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1))
        dicxx = list(np.squeeze(self.model.predict(resized_image)))
        print(dicxx)
        num = dicxx.index(max(dicxx))
        answ = copy.deepcopy(self.LABEL_DICT_CH[list(self.LABEL_DICT.keys())[list(self.LABEL_DICT.values()).index(num)]])
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
