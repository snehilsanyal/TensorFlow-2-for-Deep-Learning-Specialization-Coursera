import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
	Dense(64, activation = 'elu', input_shape = (32,)),
	Dense(100, activation = 'softmax')
	])

model.compile(
	optimizer = 'rmsprop',
	loss = 'categorical_crossentropy',
	metrics = ['accuracy']
	)

# X_train (num_samples, num_features)
# y_train (num_samples, num_classes) if y_train is a one-hot encoded vector of length num_classes
# y_train (num_samples,) if y_train is just a single integer (class)
# We should use categorical or sparse_categorical corresponding to one-hot encoded or single integer versions

# If you want to train model for one epoch (one pass through the dataset)
model.fit(X_train, y_train)
# For more than one epochs
model.fit(X_train, y_train, epochs = 10)
# Generating Batch Size, default 32
model.fit(X_train, y_train, epochs = 10, batch_size = 16)

# Returns a TensorFlow History Object
history = model.fit(X_train, y_train, epochs = 10, batch_size = 16)
