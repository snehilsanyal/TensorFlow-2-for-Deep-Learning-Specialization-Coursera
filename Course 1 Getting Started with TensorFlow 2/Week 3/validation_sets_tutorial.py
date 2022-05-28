import tensorflow as tf 

from sklearn.datasets import load_diabetes
diabetes_dataset = load_diabetes()
# diabetes_dataset is a Dictionary with keys
print(diabetes_dataset['DESCR'])
print(diabetes_dataset.keys())

data = diabetes_dataset['data']
targets = diabetes_dataset['target']

## Normalize the target data
targets = (targets - targets.mean(axis = 0))/targets.std()

## Split data
# Split the data into train and test sets
from sklearn.model_selection import train_test_split
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size = 0.1) 

print(train_data.shape, test_data.shape, train_targets.shape, test_targets.shape)

## Train a feed forward model
# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_model():
    model = Sequential([
        Dense(128, activation = 'relu', input_shape = (train_data.shape[1],)),
        Dense(128, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(1)
    ])
    return model
# Instantiate model
model = get_model()

# Print the model summary

model.summary()

# Compile the model

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

# Train the model, with some of the data reserved for validation

history = model.fit(train_data, train_targets, epochs = 100, validation_split = 0.15, batch_size = 64, verbose = 2)

# Evaluate the model on the test set

model.evaluate(test_data, test_targets, verbose = 2)

import matplotlib.pyplot as plt
#%matplotlib inline

print(history.history.keys())
# Plot the training and validation mae

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('MAE vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(['Training', 'Validation'], loc = 'upper right')
plt.show()

# Plot the training and validation loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

