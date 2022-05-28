import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax

# Feedforward Neural Network

model = Sequential([
	Flatten(input_shape = (28, 28)),
	Dense(16, activation = 'relu', name = 'layer_1'),
	Dense(16, activation = 'relu', name = 'layer_2'),
	Dense(10, name = 'layer_3'),
	Softmax()
	])

print(model.summary())