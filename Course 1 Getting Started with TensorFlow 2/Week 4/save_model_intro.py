from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
	Dense(16, activation = 'elu'),
	Dropout(0.3),
	Dense(3, activation = 'softmax')])
model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy',
	metrics = ['acc','mae'])
# Set the save_weights_only argument to False  to save the whole model
checkpoint = ModelCheckpoint(filepath = 'my_model', save_weights_only = False)
model.fit(X_train, y_train, epochs = 10, callbacks = [checkpoint])

## Once we save the model every epoch, the files will change
# my_model/assets
# my_model/saved_model.pb
# my_model/variables/variables.data-00000-of-00001
# my_model/variables/variables.index

## Assets subdirectory stores files which are required for constructing
## TensorFlow Graphs

## The variables folder contains saved weights of  the model

## The file saved_model.pb is the file that stores the TensorFlow Graph itself
## We can think of this as the saved model itself which contains when we built
## compiled the model, and the optimizer state which is needed if we want to 
## resume the training from a saved model.

## Another approach
checkpoint = ModelCheckpoint('keras_model.h5', save_weights_only = False)

# keras_model.h5 saves all the model weights and architecture

model.fit(X_train, y_train, epochs = 10)
# This saves the model in TensorFlow Native format
model.save('my_model')

## SInce we are saving the whole model we dont need the previous code for
## defining the architecture agian
from tensorflow.keras.models import load_model
new_model = load_model('my_model')
new_keras_model = load_model('keras_model.h5')
