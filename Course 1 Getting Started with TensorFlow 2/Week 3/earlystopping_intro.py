## We will use callback to implement another type of regularization called early stopping
## Early stopping is a technique that monitors the performance of a network for every
## epoch on a validation set during the training run and terminates the training
## conditional on the validation performance.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
	Conv1D(16, 5, activation = 'relu', input_shape = (128, 1),
	MaxPooling1D(4),
	Flatten(),
	Dense(10, activation = 'softmax'))])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
early_stopping = EarlyStopping()
model.fit(X_train, y_train, validation_split = 0.2, epochs = 100, callbacks = [early_stopping])

# Which parameter does the early stopping callback monitor? For this we have a keyword argument...

early_stopping = EarlyStopping(monitor = 'val_loss')
# THe value of the argument is checked from the history object dictionary keys
early_stopping = EarlyStopping(monitor = 'val_accuracy')

# As soon as the performance measure gets worse from one epoch to next, the training is terminated

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
## We also have a min_delta argument which is the minimum value to be counted as an improvement 
## in the validation performance, and hence not changing the patience. If min_delta is 0.001
## then the callback will count it as no improvement and patience counter will be increased by 1

## The mode argument tells the early stopping callback the direction in which we want the 
## monitored argument to change (increase or decrease). Default value of mode is auto.
## In this case we want the val_loss to increase so mode is max
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, min_delta = 0.01, mode = 'max')
