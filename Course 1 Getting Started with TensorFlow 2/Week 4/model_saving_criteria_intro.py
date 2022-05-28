from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
	Dense(16, activation = 'relu'),
	Dropout(0.3),
	Dense(3, activation = 'softmax')])
model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy',
	metrics = ['acc', 'mae'])
## All these checkpoints replace the previous model checkpoint file
# Argument save_freq tells you when to save the checkpoint
checkpoint = ModelCheckpoint(filepath = 'training_run_1/my_model', save_weights_only = True,
	save_freq = 'epoch')
# Another approach, save_freq = 1000 means the model checkpoint will be saved 
# after seeing 1000 samples from last, depending on the batch size every 1000%16 iterations
# the model checkpoint will be saved
checkpoint = ModelCheckpoint(filepath = 'training_run_1/my_model', save_weights_only = True,
	save_freq = 1000)
# Another approach, the model will save weights only according to the performance metric criteria
# So the model saves checkpoint depending on whether the val_loss is the best, as far as training
# is concerned
checkpoint = ModelCheckpoint(filepath = 'training_run_1/my_model', save_weights_only = True,
	save_best_only = True, monitor = 'val_loss')
# We can also change the monitor argument to val_acc and save model checkpoints only when
# the best val_acc occurs
checkpoint = ModelCheckpoint(filepath = 'training_run_1/my_model', save_weights_only = True,
	save_best_only = True, monitor = 'val_acc')
# Another argument can be added, mode
checkpoint = ModelCheckpoint(filepath = 'training_run_1/my_model', save_weights_only = True,
	save_best_only = True, monitor = 'val_acc', mode = 'max')


## Finally, the filename can also be formatted in a way such that the keys from the logs
## dictionary are included
## This way every 1000 samples, a new file will be created depending on the epoch and batch number
checkpoint = ModelCheckpoint(filepath = 'training_run_1/my_model.{epoch}.{batch}', save_weights_only = True,
	save_best_only = True, monitor = 'val_acc', mode = 'max')
## These files will not be replaced by previous files
checkpoint = ModelCheckpoint(filepath = 'training_run_1/my_model.{epoch}-{val_loss:.4f}',
	save_weights_only = True)

# Saves model checkpoint at the end of each epoch 
model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 10,
	batch_size = 16, callbacks = [checkpoint])

