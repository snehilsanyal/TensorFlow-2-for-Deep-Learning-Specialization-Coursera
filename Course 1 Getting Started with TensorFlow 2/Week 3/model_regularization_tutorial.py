## Adding regularization with weight decay and dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import regularizers


def get_regularized_model(weight_decay, dropout_rate):
	model = Sequential([
		Dense(128, kernel_regularizer = regularizers.l2(weight_decay),
			activation = 'relu', input_shape = (train_data.shape[1], )),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(weight_decay), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(weight_decay), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(weight_decay), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(weight_decay), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(weight_decay), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(1)
		])
	return model

