from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D

# In the models we have worked with so far, we have not specified the initial values of the weights and biases in each layer of our neural networks.
# The default values of the weights and biases in TensorFlow depend on the type of layers we are using.
# For example, in a Dense layer, the biases are set to zero (zeros) by default, while the weights are set according to glorot_uniform, the Glorot uniform initialiser.
# The Glorot uniform initialiser draws the weights uniformly at random from the closed interval  [−𝑐,𝑐] , where

# Initialising your own weights and biases
# We often would like to initialise our own weights and biases, and TensorFlow makes this process quite straightforward.
# When we construct a model in TensorFlow, each layer has optional arguments kernel_initialiser and bias_initialiser, which are used to set the weights and biases respectively.
# If a layer has no weights or biases (e.g. it is a max pooling layer), then trying to set either kernel_initialiser or bias_initialiser will throw an error.
# Let's see an example, which uses some of the different initialisations available in Keras.

model = Sequential([
	Conv1D(16, kernel_size = 3, input_shape = (128,64), kernel_initializer = 'random_uniform', bias_initializer = 'zeros', activation = 'relu'),
	MaxPooling1D(pool_size = 4),
	Flatten(),
	Dense(64, kernel_initializer = 'he_uniform', bias_initializer = 'ones', activation = 'relu')
	])

# Custom weight and bias initialisers
# It is also possible to define your own weight and bias initialisers. Initializers must take in two arguments, the shape of the tensor to be initialised, and its dtype.
# Here is a small example, which also shows how you can use your custom initializer in a layer.

import tensorflow.keras.backend as K 

def custom_init(shape, dtype = None):
	return K.random_normal(shape, dtype = None)

model.add(Dense(64, kernel_initializer = custom_init))
print(model.summary())

# Visualising the initialised weights and biases¶
# Finally, we can see the effect of our initialisers on the weights and biases by plotting histograms of the resulting values. Compare these plots with the selected initialisers for each layer above.

import matplotlib.pyplot as plt 
fig, axes = plt.subplots(5, 2, figsize=(12,16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Filter out the pooling and flatten layers, that don't have any weights
weight_layers = [layer for layer in model.layers if len(layer.weights) > 0]

for i, layer in enumerate(weight_layers):
    for j in [0, 1]:
        axes[i, j].hist(layer.weights[j].numpy().flatten(), align='left')
        axes[i, j].set_title(layer.weights[j].name)