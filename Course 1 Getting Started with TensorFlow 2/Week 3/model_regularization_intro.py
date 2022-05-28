from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Binary Classification Model

model = Sequential([
	Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
	# The argument of l2 regularizer is the coefficient mulitplied by sum of squared weights in the layer
	Dense(1, activation = 'sigmoid')
	])

# The overall loss function will be the binary cross entropy function 
# added with sum of squared weights of first layer multiplied by weight decay coefficient 0.001
# Why regularization? The sum of squared weights term, large weights are penalized using this term
# which leads to finding a simpler function that fits the data (not a complex function or no overfitting)

model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(inputs, targets, validation_split = 0.25)

## We can also use l1 regularization instead of l2. l1 regularization is sum of absolute weight values
# instead of sum of squared weight values

model = Sequential([
	Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(0.005)),
	Dense(1, activation = 'sigmoid')
	])

# l1 regularizer pacifies the network weights by putting some of the weights to 0
# Both can be used as well
model = Sequential([
	Dense(64, activation = 'relu', 
		kernel_regularizer = tf.keras.regularizers.l1_l2(l1 = 0.005, l2 = 0.001)),
	Dense(1, activation = 'sigmoid')
	])
## Usually we only apply weight regularization to first layer, Dense or Conv (weight matrix, kernels)
## We can also regularize biases as well, which will add another term to the loss function in terms of bias

model = Sequential([
	Dense(64, activation = 'relu', 
		kernel_regularizer = tf.keras.regularizers.l1_l2(l1 = 0.005, l2 = 0.001),
		bias_regularizer = tf.keras.regularizers.l2(0.001)),
	Dense(1, activation = 'sigmoid')
	])
## Dropout also has a regularizing effect 
# Each weight connection between the 2 Dense layers are set to 0 with a probability 0.5
# Inherently, this is done by multiplying a Bernoulli RV to the weights that is why
# this is known as Bernoulli Dropout
# Each weight is randomly dropped out independently from one another, Dropout has also
# been applied to each element in the batch independently at training time

model = Sequential([
	Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1_l2(l1 = 0.005, l2 = 0.001)),
	Dropout(0.5),
	Dense(1, activation = 'sigmoid')
	])

model.fit(inputs, targets, validation_split = 0.25)	# Training mode, Dropout
model.evaluate(val_inputs, val_targets)				# Testing mode, No Dropout
model.predict(test_inputs)							# Testing mode, No Dropout

# The lower level control of the model train and test modes can be taken in hand
