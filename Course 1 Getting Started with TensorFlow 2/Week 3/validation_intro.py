import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(128, activation = 'tanh'))
model.add(Dense(2))

opt = Adam(learning_rate = 0.005)
model.compile(optimizer = opt, loss = 'mse', metrics = ['mape'])

# Inputs shape: (num_samples, num_features)
# Targets shape: (num_samples, 2)

history = model.fit(inputs, targets, validation_split = 0.2)
print(history.history.keys()) # dict_keys(['loss', 'mape', 'val_loss', 'val_mape'])

## Another approach
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
model.fit(X_train, y_train, validation_data = (X_test, y_test))

## Yet another approach
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1)
model.fit(X_train, y_train, validation_data = (X_val, y_val))

