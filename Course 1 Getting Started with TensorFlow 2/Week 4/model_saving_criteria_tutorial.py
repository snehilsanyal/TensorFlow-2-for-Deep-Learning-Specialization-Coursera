# Import dataset part from previous scripts and 2 useful functions

from tensorflow.keras.callbacks import ModelCheckpoint
# Create a TensorFlow checkpoint to save model weights, with epoch and batch
checkpoint_5000_path = 'model_checkpoints_5000/checkpoint_{epoch:02d}_{batch:04d}'
checkpoint_5000 = ModelCheckpoint(filepath = checkpoint_5000_path, save_weights_only = True,
	save_freq = 5000, verbose = 1)

## Fit the model with the created checkpoint
model = get_new_model()
model.fit(x_train, y_train, epochs = 3, validation_data = (x_test, y_test),
	batch_size = 10, callbacks = [checkpoint_5000], verbose = 1)

## Work with model saving criteria

# Use tiny training and test set -- will overfit!

x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]

# Create a new instance of untrained model

model = get_new_model()

# Create Tensorflow checkpoint object which monitors the validation accuracy

checkpoint_best_path = 'model_checkpoints_best/checkpoint'
checkpoint_best = ModelCheckpoint(filepath = checkpoint_best_path, 
                                 save_weights_only = True,
                                  save_freq = 'epoch',
                                  save_best_only = True,
                                  monitor = 'val_accuracy',
                                 verbose =1)

# Fit the model and save only the weights with the highest validation accuracy

history = model.fit(x_train, y_train, epochs = 50, validation_data = (x_test, y_test),
                   batch_size = 10, callbacks = [checkpoint_best], verbose = 0)
## Results
# Epoch 00001: val_accuracy improved from -inf to 0.08000, saving model to model_checkpoints_best/checkpoint
# Epoch 00002: val_accuracy did not improve from 0.08000
# Epoch 00003: val_accuracy improved from 0.08000 to 0.11000, saving model to model_checkpoints_best/checkpoint
# Epoch 00004: val_accuracy did not improve from 0.11000
# Epoch 00005: val_accuracy did not improve from 0.11000
# Epoch 00006: val_accuracy improved from 0.11000 to 0.13000, saving model to model_checkpoints_best/checkpoint
# Epoch 00007: val_accuracy did not improve from 0.13000
# Epoch 00008: val_accuracy did not improve from 0.13000
# Epoch 00009: val_accuracy did not improve from 0.13000
# Epoch 00010: val_accuracy improved from 0.13000 to 0.19000, saving model to model_checkpoints_best/checkpoint

# Plot training and testing curves

import pandas as pd

df = pd.DataFrame(history.history)
df.plot(y=['accuracy', 'val_accuracy'])

## Check loading weights to a new model should lead to same result
# Create a new model with the saved weights

new_model = get_new_model()
new_model.load_weights(checkpoint_best_path)
get_test_accuracy(new_model, x_test, y_test) 
