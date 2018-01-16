# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:35:26 2018

@author: Arjun
"""

import pandas as pd
import numpy as np
import seaborn as sb 
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
#Step1- Playing with data
#Getting the data
train = pd.read_csv('C:/Users/Arjun/Desktop/Project december/Kaggle/MNIST/train.csv')
test =  pd.read_csv('C:/Users/Arjun/Desktop/Project december/Kaggle/MNIST/test.csv')
print(train)
print(test)
#Checking for Null values if any
train.isnull().any()
test.isnull().any()
#there aren't any null values, Next we can check the occurances of the digits
plot = sb.countplot(train["label"])
train["label"].value_counts()
#Normalizing data, We can remove the label column and save it in a separate variable
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
test = test / 255.0
#Reshape the data
X_train = X_train.values.reshape(-1,28,28,1)#28,28 image witdth and height, 1 color channel for monochrome images,-1 batch size 
test = test.values.reshape(-1,28,28,1)#batch size can be any number arbitrarily.
#one hot encoding can be created with the labels, we can use to_categorical from keras.utils
Y_train = to_categorical(Y_train, num_classes = 10)
#step-2 setting up the neural network
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
#CNN
model = Sequential()
#first convolution filter with patch size 5,5 no of channels 1, output channels =32
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
#convolution followed by ReLu 
#padding = 'Same' points out that the output image size should be same as the input
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
#Performing MaxPooling which reduces the size by the factor of 2,2
model.add(MaxPool2D(pool_size=(2,2)))
#dropout percentage
model.add(Dropout(0.25))
#second convolution filter patch size =3,3 
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
#flatten phase
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
#Loss and optimiser 
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#Training the model
print("Training the model")
#if split is perfromed already use validation data or else you can use validation split
model.fit(X_train, Y_train, nb_epoch=5, batch_size=86,validation_data = (X_val,Y_val), verbose=1)
predictions = model.predict(test)
# selecting the index with max probability
predictions = np.argmax(predictions,axis = 1)
predictions = pd.Series(predictions,name="Label")
final_predictions= pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)