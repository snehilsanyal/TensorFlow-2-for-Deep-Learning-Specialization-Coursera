from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Dense 64 is hidden layer, Dense 10 is output layer, We didnot tell model
# about the input layer, which can be done just before initializing different
# biases and weights before training

model_1 = Sequential([Dense(64, activation = 'relu'), Dense(10, activation = 'softmax')])

# Another definition
model_2 = Sequential([Dense(64, activation = 'relu', input_shape = (784, )), Dense(10, activation = 'softmax')])

# Yet another way
model_3 = Sequential()

model_3.add(Dense(64, activation = 'relu', input_shape = (784, )))
model_3.add(Dense(10, activation = 'softmax'))

model_4 = Sequential([Flatten(input_shape = (28, 28)), Dense(64, activation = 'relu'), Dense(10, activation = 'softmax')])
