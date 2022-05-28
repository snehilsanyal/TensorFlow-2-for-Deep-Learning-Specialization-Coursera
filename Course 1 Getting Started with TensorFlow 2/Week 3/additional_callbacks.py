
## Import from previous scripts
import tensorflow as tf 

from sklearn.datasets import load_diabetes
diabetes_dataset = load_diabetes()

data = diabetes_dataset['data']
targets = diabetes_dataset['target']

from sklearn.model_selection import train_test_split
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size = 0.1)

## Build model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model = Sequential([
	Dense(128, activation = 'relu', input_shape = (train_data.shape[1], )),
	Dense(64, activation = 'relu'),
	Dense(64, activation = 'relu'),
	Dense(64, activation = 'relu'),
	Dense(1)])

# Compile the model 

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse', 'mae'])

# In-built Learning Rate Scheduler

# Usage: tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
# The learning rate scheduler that we implemented in the previous reading as a custom callback is also available as a built in callback.
# As in our custom callback, the LearningRateScheduler in Keras takes a function schedule as an argument.
# This function schedule should take two arguments:
# The current epoch (as an integer), and
# The current learning rate,
# and return new learning rate for that epoch.
# The LearningRateScheduler also has an optional verbose argument, which prints information about the learning rate if it is set to 1.
# Let's see a simple example

# Define the learning rate scheduler function

def lr_function(epoch, lr):
	if epoch%2==0:
		return lr
	else:
		return lr + epoch/1000

## Train the model

history = model.fit(train_data, train_targets, epochs = 10,
	callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_function, verbose =1)],
	verbose = False)

## We can also use a lambda function for scheduler

history = model.fit(train_data, train_targets, epochs = 10,
	callbacks = [tf.keras.callbacks.LearningRateScheduler(lambda x: 1/(3+5*x), verbose = 1)])

# ## CSV Logger
# Usage tf.keras.callbacks.CSVLogger(filename, separator=',', append=False)
# This callback streams the results from each epoch into a CSV file. The first line of the CSV file will be the names of pieces of information recorded on each subsequent line, beginning with the epoch and loss value. The values of metrics at the end of each epoch will also be recorded.
# The only compulsory argument is the filename for the log to be streamed to. This could also be a filepath.
# You can also specify the separator to be used between entries on each line.
# The append argument allows you the option to append your results to an existing file with the same name. This can be particularly useful if you are continuing training.
# Let's see an example.

## Train the same model with CSV Logger

history = model.fit(train_data, train_targets, epochs = 10,
	callbacks = [tf.keras.callbacks.CSVLogger('results.csv'), verbose = False])

## Load the csv
import pandas as pd 
df = pd.read_csv('results.csv', index_col = 'epoch')
print(df.head())

# Lambda callbacks
# Usage tf.keras.callbacks.LambdaCallback(
#         on_epoch_begin=None, on_epoch_end=None, 
#         on_batch_begin=None, on_batch_end=None, 
#         on_train_begin=None, on_train_end=None)

# Lambda callbacks are used to quickly define simple custom callbacks with the use of lambda functions.

# Each of the functions require some positional arguments.

# on_epoch_begin and on_epoch_end expect two arguments: epoch and logs,
# on_batch_begin and on_batch_end expect two arguments: batch and logs and
# on_train_begin and on_train_end expect one argument: logs.
# Let's see an example of this in practice.

# Print the number of epoch at the beginning of each epoch

epoch_callback = tf.keras.callbacks.LambdaCallback(
	on_epoch_begin = lambda epoch, logs: print("Starting epoch: {}").format(epoch+1))
# Print the loss at the end of each batch

batch_callback = tf.keras.callbacks.LambdaCallback(
	on_batch_begin = lambda batch, logs: print("After batch {}, the loss is {:7.2f}").format(batch, logs['loss']))

# Inform the training is finished
train_finish_callback = tf.keras.callbacks.LambdaCallback(
	on_train_end = lambda logs: print("Training finished!"))

history = model.fit(train_data, train_targets, epochs = 10, 
	callbacks = [epoch_callback, batch_callback, train_finish_callback], verbose = False)

# Reduce learning rate on plateau¶
# Usage tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss', 
#             factor=0.1, 
#             patience=10, 
#             verbose=0, 
#             mode='auto', 
#             min_delta=0.0001, 
#             cooldown=0, 
#              min_lr=0)

# The ReduceLROnPlateau callback allows reduction of the learning rate when a metric has stopped improving. The arguments are similar to those used in the EarlyStopping callback.

# The argument monitor is used to specify which metric to base the callback on.
# The factor is the factor by which the learning rate decreases i.e., new_lr=factor*old_lr.
# The patience is the number of epochs where there is no improvement on the monitored metric before the learning rate is reduced.
# The verbose argument will produce progress messages when set to 1.
# The mode determines whether the learning rate will decrease when the monitored quantity stops increasing (max) or decreasing (min). The auto setting causes the callback to infer the mode from the monitored quantity.
# The min_delta is the smallest change in the monitored quantity to be deemed an improvement.
# The cooldown is the number of epochs to wait after the learning rate is changed before the callback resumes normal operation.
# The min_lr is a lower bound on the learning rate that the callback will produce.
# Let's examine a final example.

# Train the model with the ReduceLROnPlateau callback

history = model.fit(train_data, train_targets, epochs=100, batch_size=100,
                    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="loss",factor=0.2, verbose=1)], verbose=False)

# Further reading and resources
# https://keras.io/callbacks/
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback