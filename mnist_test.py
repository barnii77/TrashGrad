from keras.datasets import mnist
import random
from Dense import Dense
from Tensor import Tensor
from Convolutional import Convolutional2D
from Activation import relu, tanh, sigmoid
from Sequential import Sequential
import numpy as np
import matplotlib.pyplot as plt


(train_X, train_y), (test_X, test_y) = mnist.load_data()
print(train_X.shape)


EPOCHS = 1000

model = Sequential([
    Dense(2, 3),
    tanh,
    Dense(3, 1),
    sigmoid
    #Convolutional2D(1, 1, 1, 1, 1, 9, 1)
])
