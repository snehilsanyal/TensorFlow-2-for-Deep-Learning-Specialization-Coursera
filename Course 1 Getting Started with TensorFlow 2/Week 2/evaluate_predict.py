import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(1, activation = 'sigmoid', input_shape = (12, ))])
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy', 'mae'])

model.fit(X_train, y_train)

# loss, accuracy = model.evaluate(X_test, y_test)
loss, accuracy, mae = model.evaluate(X_test, y_test)

# X_sample (num_samples, 12)
# X_sample (1, 12) pred (1, )
# X_sample (2, 12) pred (2, )
pred = model.predict(X_sample) # [[0.077],
							   #  [0.945]]


model = Sequential([Dense(3, activation = 'softmax', input_shape = (12, ))])
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy', 'mae'])

model.fit(X_train, y_train)
loss, accuracy, mae = model.evaluate(X_test, y_test)

# X_sample (2, 12) pred (2, 3)
pred = model.predict(X_sample) # [[0.023, 0.345, 0.625],
							   #  [0.341, 0.567, 0.123]]
							   



