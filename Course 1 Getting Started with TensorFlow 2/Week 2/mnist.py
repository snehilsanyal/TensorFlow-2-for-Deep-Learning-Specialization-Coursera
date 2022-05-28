import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

# Create a CNN with 16 filters of 3x3 size, input_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(16, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))

# Add Maxpooling layer with pool size 3x3
model.add(MaxPooling2D(pool_size = (3, 3)))

# Add Flatten() layer
model.add(Flatten())

# Add Dense layer with 10 units and softmax activation
model.add(Dense(10, activation = 'softmax'))

# Get model summary
print(model.summary)

# Compile Model
opt = tf.keras.optimizers.Adam(learning_rate = 0.005)
acc = tf.keras.metrics.SparseCategoricalCrossentropy()
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(
	optimizer = 'adam',
	loss = 'sparse_categorical_crossentropy',
	metrics = ['acc', 'mae']
	)

# Visualize loss, optimizer and metrics
print(model.optimizer)
print(model.loss)
print(model.metrics)
print(model.optimizer.lr)

## Import Data and Fit Model
from tensorflow.keras.preprocessing import Image
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

# Load data
fashion_mnist_data = tf.keras.datasets.fashion_mnist 
(train_images, train_labels), (test_images, test_labels) = fasion_mnist_data.load_data() 

# Print the shape of the training data
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape) 

# Define the labels

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# Rescale all images to 0 to 1
train_images = train_images/255.
test_images = test_images/255.

## Sanity Check
i = 0
img = train_images[i, :, :]
plt.imshow(img)
plt.show()
print("Label: {}".format(labels[train_labels[i]]))

# Fit model with only 2 epochs, train_images needs a dummy axis
model.fit(train_images[..., np.newaxis], train_labels, epochs = 2, batch_size = 256)

# Store as a history object, Verbose = 0, shows nothing, 1, shows each sample in each epoch, 2, shows results per epoch
history = model.fit(train_images[..., np.newaxis], train_labels, epochs = 8, batch_size = 256, verbose = 2)

# history.history is a dictionary
print(history.history)

# Load the history into a pandas Dataframe
df = pd.DataFrame(history.history)
df.head()

# Make a plot for the loss
loss_plot = df.plot(y = 'loss', title = 'Loss vs Epochs', legend = False)
loss_plot.set(xlabel = 'Epochs', ylabel = 'Loss')

# Make a plot for the accuracy

acc_plot = df.plot(y = 'sparse_categorical_crossentropy', title = 'Accuracy vs Epochs', legend = False)
acc_plot.set(xlabel = 'Epochs', ylabel = 'Accuracy')

# Make a plot for the additional metric

mae_plot = df.plot(y = 'mean_absolute_error', title = 'MAE vs Epochs', legend = False)
mae_plot.set(xlabel = 'Epochs', ylabel = 'MAE')

# Evaluate the model on test set
test_loss, test_accuracy, test_mae = model.evaluate(test_images[..., np.newaxis], test_labels, verbose = 2)

# Make predictions on the model
random_index = np.random.choice(test_images.shape[0])

# Get a random test image from test_images
test_image = test_images[random_index]
plt.imshow(test_image)
plt.show()
print("Label : {}".format(labels[test_labels[random_index]]))

# Get the model predictions
predictions = model.predict(test_image[np.newaxis, ..., np.newaxis])
print("Labels : {}".format(labels[np.argmax(predictions)]))