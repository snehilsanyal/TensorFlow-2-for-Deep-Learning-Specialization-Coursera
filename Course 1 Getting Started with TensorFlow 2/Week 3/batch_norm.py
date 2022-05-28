import tensorflow as tf 
from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()
from sklearn.model_selection import train_test_split

data = diabetes_dataset['data']
targets = diabetes_dataset['target']

## Normalize Labels
# Normalise the target data (this will make clearer training curves)

targets = (targets - targets.mean(axis = 0))/(targets.std())

# Split the dataset

train_data, train_labels, test_data, test_labels = train_test_split(data, targets, test_size = 0.1)

# We can implement batch normalisation into our model by adding it in the same way as any other layer.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout

# Build the model

model = Sequential([
	Dense(64, input_shape = (train_data.shape[1], ), activation = 'relu'),
	BatchNormalization(),
	Dropout(0.5),
	BatchNormalization(),
	Dropout(0.5),
	Dense(256, activation = 'relu')
	])
print(model.summary())


# Recall that there are some parameters and hyperparameters associated with batch normalisation.
# The hyperparameter momentum is the weighting given to the previous running mean when re-computing it with an extra minibatch. By default, it is set to 0.99.
# The hyperparameter  𝜖  is used for numeric stability when performing the normalisation over the minibatch. By default it is set to 0.001.
# The parameters  𝛽  and  𝛾  are used to implement an affine transformation after normalisation. By default,  𝛽  is an all-zeros vector, and  𝛾  is an all-ones vector.

# Customising parameters
# These can all be changed (along with various other properties) by adding optional arguments to tf.keras.layers.BatchNormalization().
# We can also specify the axis for batch normalisation. By default, it is set as -1.
# Let's see an example.

## Add a customized BatchNormalization layer

model.add(tf.keras.layers.BatchNormalization(
	momentum = 0.95,
	epsilon = 0.005,
	axis = -1,
	beta_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05),
	gamma_initializer = tf.keras.initializers.Constant(value = 0.9))
	)

# Add output layer
model.add(Dense(1))

# Let's now compile and fit our model with batch normalisation, and track the progress on training and validation sets.
# First we compile our model.

model.compile(optimizer = 'adam',
	loss = 'mse',
	metrics = ['mae'])

# Train the model
history = model.fit(train_data, train_targets, epochs = 100, validation_split = 0.15, batch_size = 64, verbose = False)

# Plot the learning curves

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline 

df = pd.DataFrame(history.history)
print(df.head())
epochs = np.arange(len(df))

fig = plt.figure(figsize = (12, 4))

# Loss Plot
ax = fig.add_subplot(121)
ax.plot(epochs, df['loss'], label = 'Train')
ax.plot(epochs, df['val_loss'], label = 'Validation')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss vs Epochs')
ax.legend()

ax = fig.add_subplot(122)
ax.plot(epochs, df['mae'], label = 'Train')
ax.plot(epochs, df['val_mae'], label = 'Validation')
ax.set_xlabel('Epochs')
ax.set_ylabel('MAE')
ax.set_title('MAE vs Epochs')
ax.legend()

