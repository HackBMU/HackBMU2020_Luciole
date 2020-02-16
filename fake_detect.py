import cv2
import matplotlib.pyplot as plt
import pickle
import keras
from keras.layers import BatchNormalization
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout
import random
from keras.preprocessing.image import ImageDataGenerator
img_shape = 224


CATAGORIES = ["training_fake","training_real"]
training_data = []

for x in os.listdir(datadir):
  for y in os.listdir(str(datadir+'/'+x)):
    path = os.path.join(datadir, x)
    img = cv2.imread(os.path.join(path, y), 0)
    img = cv2.resize(img, (224, 224))
    training_data.append([img[..., ::-1], CATAGORIES.index(x)])
    
random.shuffle(training_data)
X, y = [], []
for features, label in training_data:
    X.append(features)
    y.append(label)
    
import numpy as np
X = np.array(X)
X = X/266.0

model = Sequential()
model.add(Conv2D(632, (3,3), input_shape=(96, 96, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), input_shape=(96, 96, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
model.fit(X, y,batch_size=32,epochs=10,validation_split=0.1)

model.fit('real-vs-fake')
