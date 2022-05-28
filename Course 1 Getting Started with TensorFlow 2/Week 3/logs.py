import tensorflow as tf 

# The logs dictionary stores the loss value, along with all of the metrics we are using at the end of a batch or epoch.
# We can incorporate information from the logs dictionary into our own custom callbacks.
# Let's see this in action in the context of a model we will construct and fit to the sklearn diabetes dataset that we have been using in this module.
# Let's first import the dataset, and split it into the training and test sets.

from sklearn.datasets import load_diabetes
diabetes_dataset = load_diabetes()
from sklearn.model_selection import train_test_split
data = diabetes_dataset['data']
targets = diabetes_dataset['target']

train_data, test_data, train_labels, test_labels = train_test_split(data, targets, test_size = 0.1)

# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import Callback 

model = Sequential([
	Dense(128, activation = 'relu', input_shape = (train_data.shape[1], )),
	Dense(64, activation = 'relu'),
	BatchNormalization(),
	Dense(64, activation = 'relu'),
	Dense(64, activation = 'relu'),
	Dense(1)
	])

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

## Define a custom Callback using logs dictionary
# Now we define our custom callback using the logs dictionary to access the loss and metric values
class LossAndMetricCallback(Callback):

	# Print the loss after every second batch of training
	def on_train_batch_end(self, batch, logs = None):
		if batch%2==0:
			print("After batch {} the loss is {:7.2f}".format(batch, logs['loss']))
	# Print the loss after every batch in the test set
	def on_test_batch_end(self, batch, logs = None):
		print("After batch {} the loss is {:7.2f}".format(batch, logs['loss']))
	# Print the loss and MAE after each epoch
	def on_epoch_end(self, epoch, logs = None):
		print("Epoch {}: Average loss is {:7.2f}, mean absolute error is {:7.2f}".format(epoch, logs['loss'], logs['mae']))
	# Notify the user when prediction has finished on each batch
    def on_predict_batch_end(self,batch, logs=None):
        print("Finished prediction on batch {}!".format(batch))

# We now fit the model to the data, and specify that we would like to use our custom callback LossAndMetricCallback()

history = model.fit(train_data, train_labels, epochs = 20, batch_size = 100, callbacks = [LossAndMetricCallback()], verbose = False)

# We can also use our callback in the evaluate function

model_eval = model.evaluate(test_data, test_labels, batch_size = 10, callbacks = [LossAndMetricCallback()], verbose = False)

## Learning Rate Scheduler
# We are going to define a callback to change the learning rate of the optimiser of a model during training. We will do this by specifying the epochs and new learning rates where we would like it to be changed.
# First we define the auxillary function that returns the learning rate for each epoch based on our schedule.

lr_schedule = [
    (4, 0.03), (7, 0.02), (11, 0.005), (15, 0.007)
]

def get_new_epoch_lr(epoch, lr):
	epoch_in_sched = [i for i in range(len(lr_schedule)) if lr_schedule[i][0]==int(epoch)]
	if len(epoch_in_sched)>0:
		return epoch_in_sched[0][1]
	else:
		return lr 

## Define the custom LR Scheduler

class LRScheduler(Callback):

	def __init__(self, new_lr):
		super(LRScheduler, self).__init__()
		# Add new learning rate function to our callback
		self.new_lr = new_lr 

	def on_epoch_begin(self, epochs, logs = None):
		if not hasattr(self.model.optimizer, 'lr'):
			raise ValueError('Error: Optimizer does not have a learning rate.')
		# Get the current learning rate
		curr_rate  = float(tf.keras.backend.get_value(self.model.optimizer.lr))
		# Get the auxillary learning rate
		scheduled_rate = self.new_lr(epoch, curr_rate)

		# Set the learning rate to the scheduled learning rate
		tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_rate)
		print("Learning rate for epochs {} is {:7.3f}".format(epoch, scheduled_rate))

# Build the same model as before

new_model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dense(64,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)        
])

# Compile the model

new_model.compile(loss='mse',
                optimizer="adam",
                metrics=['mae', 'mse'])
# Fit the model with our learning rate scheduler callback

new_history = new_model.fit(train_data, train_targets, epochs=20,
                            batch_size=100, callbacks=[LRScheduler(get_new_epoch_lr)], verbose=False)

## Read more
# https://www.tensorflow.org/guide/keras/custom_callback
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback