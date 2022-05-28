## We can save the model in 2 formats, native TF and hdf5 used by Keras
## How to save a model during training?

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
## Import the model checkpoint callback
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
	Dense(64, activation = 'sigmoid', input_shape = (10, )),
	Dense(1)])

model.compile(optimizer = 'sgd', loss = BinaryCrossentropy(from_logits = True))
## Instantiate the checkpoint object
## ModelCheckpoint constructor takes argument as the name of model
## and save_weights_only
checkpoint = ModelCheckpoint('my_model', save_weights_only = True)
model.fit(X_train, y_train, epochs = 20, callbacks = [checkpoint])

## This callback will save the model weights after each epoch
## Since the filename is same, the saved weights will get overwritten every epoch

## THese 3 files will be created 
# checkpoint
# my_model.data-00000-of-00001
# my_model.index

## Alternate approach, save as .h5 (hdf5) format used by Keras

checkpoint = ModelCheckpoint('my_model.h5', save_weights_only = True)
## This will save only one file 
# my_model.h5

## After saving the model weights in previous run, can we load them?
## Since we didnot save the architecture of the model, only weights
## We need the previous architecture code, so that we can load the weights

model = Sequential([
	Dense(64, activation = 'sigmoid', input_shape = (10, )),
	Dense(1)])
model.load_weights('my_model')

## ALternatively if we have saved the file in hdf5format 
model.load_weights('my_model.h5')


## How can we save the model weights manually without caling the model checkpoint callback

model = Sequential([
	Dense(64, activation = 'sigmoid', input_shape = (10, )),
	Dense(1)])
model.compile(optimizer = 'sgd', loss = 'mse', metrics = ['mae'])
early_stopping = EarlyStopping(monitor = 'val_mae', patience = 	2)
model.fit(X_train, y_train, validation_split = 0.2, epochs = 50,
	callbacks = [early_stopping])
model.save_weights('my_model')
