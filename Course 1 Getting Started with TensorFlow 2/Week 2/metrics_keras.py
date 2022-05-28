import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow.keras.backend as K

model = Sequential([
	Flatten(input_shape = (28, 28)),
	Dense(32, activation = 'relu'),
	Dense(32, activation = 'tanh'),
	Dense(10, activation = 'softmax')
	])

model.compile(optimizer = 'adam',
			  loss = 'sparse_categorical_crossentropy',
			  metrics = ['accuracy']
			  )

## Case 1: Binary Classification with Sigmoid Activation

# Suppose we are training a model for a binary classification problem with a sigmoid activation function (softmax activation functions are covered in the next case).
# Given a training example with input  𝑥(𝑖) , the model will output a float between 0 and 1. Based on whether this float is less than or greater than our "threshold" (which by default is set at 0.5), we round the float to get the predicted classification  𝑦𝑝𝑟𝑒𝑑  from the model.
# The accuracy metric compares the value of  𝑦𝑝𝑟𝑒𝑑  on each training example with the true output, the one-hot coded vector  𝑦(𝑖)𝑡𝑟𝑢𝑒  from our training data.

y_true = tf.constant([0.0,1.0,1.0])
y_pred = tf.constant([0.4,0.8,0.3])
accuracy = K.mean(K.equal(y_true, K.round(y_pred)))
print(accuracy)

## Case 2 - Categorical Classification
# Now suppose we are training a model for a classification problem which should sort data into  𝑚>2  different classes using a softmax activation function in the last layer.
# Given a training example with input  𝑥(𝑖) , the model will output a tensor of probabilities  𝑝1,𝑝2,…𝑝𝑚 , giving the likelihood (according to the model) that  𝑥(𝑖)  falls into each class.
# The accuracy metric works by determining the largest argument in the  𝑦(𝑖)𝑝𝑟𝑒𝑑  tensor, and compares its index to the index of the maximum value of  𝑦(𝑖)𝑡𝑟𝑢𝑒  to determine  𝛿(𝑦(𝑖)𝑝𝑟𝑒𝑑,𝑦(𝑖)𝑡𝑟𝑢𝑒) . It then computes the accuracy in the same way as for the binary classification case.
# In the backend of Keras, the accuracy metric is implemented slightly differently depending on whether we have a binary classification problem ( 𝑚=2 ) or a categorical classifcation problem. Note that the accuracy for binary classification problems is the same, no matter if we use a sigmoid or softmax activation function to obtain the output.

# Binary classification with softmax

y_true = tf.constant([[0.0,1.0],[1.0,0.0],[1.0,0.0],[0.0,1.0]])
y_pred = tf.constant([[0.4,0.6], [0.3,0.7], [0.05,0.95],[0.33,0.67]])
accuracy =K.mean(K.equal(y_true, K.round(y_pred)))
accuracy

# Binary accuracy and categorical accuracy
# The binary_accuracy and categorical_accuracy metrics are, by default, identical to the Case 1 and 2 respectively of the accuracy metric explained above.
# However, using binary_accuracy allows you to use the optional threshold argument, which sets the minimum value of  𝑦𝑝𝑟𝑒𝑑  which will be rounded to 1. As mentioned above, it is set as threshold=0.5 by default.
# Below we give some examples of how to compile a model with binary_accuracy with and without a threshold.

# Compile the model with default threshold (=0.5)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['binary_accuracy'])

# The threshold can be specified as follows

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])

## Sparse categorical accuracy
# This is a very similar metric to categorical accuracy with one major difference - the label  𝑦𝑡𝑟𝑢𝑒  of each training example is not expected to be a one-hot encoded vector, but to be a tensor consisting of a single integer. This integer is then compared to the index of the maximum argument of  𝑦𝑝𝑟𝑒𝑑  to determine  𝛿(𝑦(𝑖)𝑝𝑟𝑒𝑑,𝑦(𝑖)𝑡𝑟𝑢𝑒) .

# Two examples of compiling a model with a sparse categorical accuracy metric

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["sparse_categorical_accuracy"])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

## (Sparse) Top  𝑘 -categorical accuracy
# In top  𝑘 -categorical accuracy, instead of computing how often the model correctly predicts the label of a training example, the metric computes how often the model has  𝑦𝑡𝑟𝑢𝑒  in the top  𝑘  of its predictions. By default,  𝑘=5 .
# As before, the main difference between top  𝑘 -categorical accuracy and its sparse version is that the former assumes  𝑦𝑡𝑟𝑢𝑒  is a one-hot encoded vector, whereas the sparse version assumes  𝑦𝑡𝑟𝑢𝑒  is an integer.

# Compile a model with a top-k categorical accuracy metric with default k (=5)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["top_k_categorical_accuracy"])

# Specify k instead with the sparse top-k categorical accuracy

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)]) 

## Custom metrics
# It is also possible to define your own custom metric in Keras. You will need to make sure that your metric takes in (at least) two arguments called y_true and y_pred and then output a single tensor value.

# Define a custom metric

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

# Specify k instead with the sparse top-k categorical accuracy

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[mean_pred])

## Multiple metrics
# Finally, it is possible to use multiple metrics to judge the performance of your model.
# Here's an example:

# Compile the model with multiple metrics

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[mean_pred, "accuracy",tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)])
