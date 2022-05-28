## Callbacks are objects which are used to track loss metrics
## and take some actions based on them. We can construct our own
## custom callback using base class Callback

# Callback is the base class from which all the callbacks inherit
from tensorflow.keras.callbacks import Callback

## Create a custom Callback

class my_callback(Callback):

	def on_train_begin(self, logs = None):
		# Do something on the start of training
	def on_train_batch(self, batch, logs = None):
		# Do something on the start of every batch iteration
	def on_epoch_end(self, epoch, logs = None):
		# Do something on the end of every epoch
	## We can also do  this for evaluation and prediction callbacks

# Once done, we can pass this as an argument to model.fit

history = model.fit(X_train, y_train, epochs = 5, callbacks = [my_callback()])

# history is also a callback, which is automatically included in every training run
# whenever model.fit is called. The action this callback takes is simply the record
# of loss and accuracies and store them as a dictionary in its history attribute.

