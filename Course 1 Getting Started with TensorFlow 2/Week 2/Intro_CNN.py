from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

model = Sequential()

# 1st dimension is always None, because it is the batch_size which is felxible

## Conv2D Layer
# 16 Filters, of size 3x3 applied on 32x32 RGB Image 
# Then the output is passed through an activation of relu
# (None, 30, 30, 16)
model.add(Conv2D(16, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
## To get the padding as SAME, i.e. we want same output shape as input, can be written as:
# model.add(Conv2D(16, kernel_size = (3, 3), padding = 'SAME', activation = 'relu', input_shape = (32, 32, 3)))
# Also try changing the strides


## Add Pooling Layer
# Pooling window size of 3x3 pixels
# (None, 10, 10, 16)
model.add(MaxPooling2D((3,3)))

## Add Flatten Layer
# Takes the input and rolls it down to one-dimensional vector for further layers
# (None, 1600)
model.add(Flatten())


## Add Dense Layer
# Dense layer with 64 units and relu activation
# (None, 64)
model.add(Dense(64, activation = 'relu'))

## Add another Dense layer
# Dense layer with 10 units and softmax activation
# (None, 10)
model.add(Dense(10, activation = 'softmax'))
