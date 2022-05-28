import tensorflow as tf 

# Saving and loading model weights
# Load and inspect CIFAR-10 dataset
# The CIFAR-10 dataset consists of, in total, 60000 color images, each with one of 10 labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. For an introduction and a download, see this link.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.
x_test = x_test/255.

# Use a smaller subset
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:1000], y_test[:1000]

# Plot the first 10 CIFAR-10 images 
import matplotlib.pyplot as plt 
fig, ax = plt.subplots(1, 10, figsize = (10, 1))
for i in range(10):
	ax[i].set_axis_off()
	ax[i].imshow(x_train[i])

# Introduce function to test model accuracy
def get_test_accuracy(model, x_test, y_test):
	test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0)
	print("Accuracy :{acc:0.3f}".format(acc= test_acc))

# Introduce function that creates new instance of a CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

def get_new_model():
	model = Sequential([
		Conv2D(filters = 16, kernel_size = (3, 3), input_shape = (32, 32, 3),
			activation = 'relu', name = 'conv_1'),
		Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', name = 'conv_2'),
		MaxPooling2D(pool_size = (4, 4), name = 'pool_1'),
		Flatten(name = 'flatten'),
		Dense(units = 32, activation = 'relu', name = 'dense_1'),
		Dense(units = 10, activation = 'softmax', name = 'dense_2')])
	model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
	return model

# Create an instance of the model and show model summary

model = get_new_model()
print(model.summary())

# Check model test accuracy

print(get_test_accuracy(model, x_test, y_test))

## Train model with checkpoints
from tensorflow.keras.callbacks import ModelCheckpoint 

checkpoint_path = 'model_checkpoints/checkpoint'
# Frequency is set to epoch, which means we would like  to save checkpoints
# after each epoch
checkpoint = ModelCheckpoint(filepath = checkpoint_path, frequency = 'epoch', 
	save_weights_only = True, verbose = 1)
# At the end of every epoch it is going to rewrite the same filename

# Fit the model with simple checkpoint which saves and overwrites model weights 
# every epoch
model.fit(x_train, y_train, epochs = 3, callbacks = [checkpoint])

# total 184K
# -rw-r--r-- 1 jovyan users   77 May  7 15:51 checkpoint
# -rw-r--r-- 1 jovyan users 174K May  7 15:51 checkpoint.data-00000-of-00001
# -rw-r--r-- 1 jovyan users 2.0K May  7 15:51 checkpoint.index

#checkpoint.data-00000-of-00001 stores all the weights, and rest is metadata

# Lets check with loading weights and creating a fresh model
# New model instance
model = get_new_model()
get_test_accuracy(model, x_test, y_test)

# Load weights from pretrained model 
model.load_weights(checkpoint_path)
# This should give the same testing accuracy as previous evaluate method
get_test_accuracy(model, x_test, y_test)