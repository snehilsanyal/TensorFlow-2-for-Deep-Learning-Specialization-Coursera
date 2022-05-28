## Import rest of the things from the previous scripts till 2 usefull functions

from tensorflow.keras.callbacks import ModelCheckpoint

# The directory inside this folder is created automatically if we save the whole 
# model
checkpoint_path = 'model_checkpoint'
checkpoint = ModelCheckpoint(filepath = checkpoint_path,
	frequency = 'epoch',
	save_weights_only = False,
	verbose = 1)
model = get_new_model()
model.fit(x_train, y_train, epochs = 3, callbacks = [checkpoint])



## Get the model's test accuracy
get_test_accuracy(model, x_test, y_test)


## Create a new model, delete the old model
del model

## Reload the model from checkpoint_path
from tensorflow.keras.models import load_model
model = load_model(checkpoint_path)
## Check whether this and the previous test accuracies are same or not
get_test_accuracy(model, x_test, y_test)

## Save the same model as h5 format
## if we save only using tf native format it first creates a directory
## and then creates 3 more subdirectories

model.save('my_model.h5')
del model 
model = load_model('my_model.h5')
get_test_accuracy(model , x_test, y_test)
