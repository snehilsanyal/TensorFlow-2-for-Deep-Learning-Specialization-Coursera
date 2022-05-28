import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json
import numpy as np


# Accessing a model's configuration
# A model's configuration refers to its architecture. 
# TensorFlow has a convenient way to retrieve a model's architecture as a dictionary. 
# We start by creating a simple fully connected feedforward neural network with 1 hidden layer.

# Build the model
model = Sequential([
	Dense(32, input_shape = (32, 32, 3), activation = 'relu', name = 'dense_1'),
	Dense(10, activation = 'softmax', name = 'dense_2')])

# A TensorFlow model has an inbuilt method get_config which returns the model's architecture as a dictionary:

# Get the model config

config_dict = model.get_config()
print(config_dict)

# Creating a new model from the config
# A new TensorFlow model can be created from this config dictionary. This model will have reinitialized weights, which are not the same as the original model.

# Create a model from the config dictionary

model_same_config = tf.keras.Sequential.from_config(config_dict)

# We can check explicitly that the config of both models is the same, but the weights are not

# Check the new model is the same architecture

print('Same config:', 
      model.get_config() == model_same_config.get_config())
print('Same value for first weight matrix:', 
      np.allclose(model.weights[0].numpy(), model_same_config.weights[0].numpy()))

## Important point:
# For models that are not `Sequential` models, use `tf.keras.Model.from_config` 
# instead of `tf.keras.Sequential.from_config`.

# Other file formats: JSON and YAML
# It is also possible to obtain a model's config in JSON or YAML formats. This follows the same pattern

# Convert the model to JSON

json_string = model.to_json()
print(json_string)

# The JSON format can easily be written out and saved as a file 

# Write out JSON config file

with open('config.json', 'w') as f:
    json.dump(json_string, f)
del json_string

# Read in JSON config file again

with open('config.json', 'r') as f:
    json_string = json.load(f)

# Reinitialize the model

model_same_config = tf.keras.models.model_from_json(json_string)

# Check the new model is the same architecture, but different weights

print('Same config:', 
      model.get_config() == model_same_config.get_config())
print('Same value for first weight matrix:', 
      np.allclose(model.weights[0].numpy(), model_same_config.weights[0].numpy()))

# The YAML format is similar. The details of writing out YAML files, loading them and using them to create a new model are similar as for the JSON files, so we won't show it here.

# Convert the model to YAML

yaml_string = model.to_yaml()
print(yaml_string)

# Writing out, reading in and using YAML files to create models is similar to JSON files.

# Further reading and resources
# https://www.tensorflow.org/guide/keras/save_and_serialize#architecture-only_saving
# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model