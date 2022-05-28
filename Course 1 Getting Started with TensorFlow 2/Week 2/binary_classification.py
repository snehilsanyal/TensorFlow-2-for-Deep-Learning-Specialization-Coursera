import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
	Dense(64, activation = 'elu', input_shape = (3,1)),
	Dense(1, activation = 'sigmoid')
	])

## Define Loss function and Optimizer
# We can also set a list of metric for evaluation which we can keep track of
# 

model.compile(
	optimizer = 'sgd', 				# 'adam', 'rmsprop', 'adadelta'
	loss = 'binary_crossentropy',	# 'mean_squared_error', 'categorical_crossentropy'
	metrics = ['accuracy', 'mae']	
	)

# The Keras API gives us freedom to use either string as an option 
# or directly the reference object that the string represents
# This representation is preferred because of flexibility and defining other properties

model.compile(
	optimizers = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9, nesterov = True), # SGD Object from tf.keras.optimizers module
	loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
	metrics = [tf.keras.metrics.BinaryAccuracy(threshold = 0.7), tf.keras.metrics.MeanAbsoluteError()]
	)

## We can also do this by initializing an object before hand

opt = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9, nesterov = True)
los = tf.keras.losses.BinaryCrossentropy(from_logits = True)
mt = [tf.keras.metrics.BinaryAccuracy(threshold = 0.7), tf.keras.metrics.MeanAbsoluteError()]
model.compile(optimizers = opt,
	loss = los,
	metrics = mt
	)

## from_logits = True, now we have to change the activation of final Dense layer
## to linear, which is the logit value just before it is squeezed to sigmoid
