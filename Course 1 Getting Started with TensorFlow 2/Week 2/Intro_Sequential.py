from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(64, activation = 'relu'), Dense(10, activation = 'softmax')])