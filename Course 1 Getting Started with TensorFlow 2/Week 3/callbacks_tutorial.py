## Callbacks are for getting feedback from the model execution and passed
## during model.fit 
from validation_sets_tutorial import train_data, train_targets, test_data, test_targets
from tensorflow.keras.callbacks import Callback 

class TrainingCallback(Callback):

	def on_begin_training(self, logs = None):
		print("Training started")
	def on_begin_epoch(self, epoch, logs = None):
		print(f"Starting epoch {epoch}")
	def on_begin_batch(self, batch, logs = None):
		print(f"Training: Starting batch {batch}")
	def on_end_epoch(self, epoch, logs = None):
		print(f"Finished epoch {epoch}")
	def on_end_batch(self, batch, logs = None):
		print(f"Finished training batch {batch}")
	def on_end_training(self, logs = None):
		print(f"Finished Training!")

def get_regularized_model(weight_decay, dropout_rate):
	model = Sequential([
		Dense(128, kernel_regularizer = regularizers.l2(wd),
			activation = 'relu', input_shape = (train_data.shape[1], )),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(wd), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(wd), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(wd), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(wd), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(128, kernel_regularizer = regularizers.l2(wd), activation = 'relu'),
		Dropout(dropout_rate),
		Dense(1)
		])
	return model

model = get_regularized_model(1e-5, 0.3)
# Sample callback passed to model.fit
history = model.fit(train_data, train_targets, epochs = 3, batch_size = 128, verbose = False,
	callbacks = [TrainingCallback()])

## We can also write callbacks for evaluate and predict methods

class EvaluateCallback(Callback):

	def on_test_begin(self, logs = None):
		print("Starting Testing ...")
	def on_test_batch_begin(self, batch, logs = None):
		print(f"Testing: Started Batch {batch}")
	def on_test_batch_end(self, batch, logs = None):
		print(f"Finished Batch {batch}")
	def on_test_end(self, logs = None):
		print("Finished Testing!")

model.evaluate(test_data, test_targets, batch_size = 128, verbose = False,
	callbacks = [EvaluateCallback()])


## Predict Callback
class PredictCallback(Callback):

	def on_predict_begin(self, logs = None):
		print("Starting Predicting ...")
	def on_predict_batch_begin(self, batch, logs = None):
		print(f"Predicting: Started Batch {batch}")
	def on_predict_batch_end(self, batch, logs = None):
		print(f"Finished Batch {batch}")
	def on_predict_end(self, logs = None):
		print("Finished Predictions!")

model.predict(test_data, verbose = False, callbacks = [PredictCallback()])

