## Import everything from the previous scripts Just writing the part of the early_stopping
## from Notebook

## First train the unregularized model

# Re-train the unregularised model

unregularized_model = get_model()
unregularized_model.compile(optimizer = 'adam', loss = 'mse')
unreg_history = unregularized_model.fit(train_data, train_targets, epochs = 100,
                                        validation_split = 0.15, batch_size = 64, verbose = False,
                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience = 2)])
# Evaluate the model on the test set

unregularized_model.evaluate(test_data, test_targets, verbose = 2)

# Re-train the regularised model

regularized_model = get_regularized_model(1e-8, 0.2)
regularized_model.compile(optimizer = 'adam', loss = 'mse')
reg_history = regularized_model.fit(train_data, train_targets, epochs = 100,
                     validation_split = 0.15, batch_size = 64, verbose = False,
                     callbacks = [tf.keras.callbacks.EarlyStopping(patience = 2)])
# Evaluate the model on the test set

regularized_model.evaluate(test_data, test_targets, verbose = False)

# Plot the training and validation loss

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 5))

fig.add_subplot(121)

plt.plot(unreg_history.history['loss'])
plt.plot(unreg_history.history['val_loss'])
plt.title('Unregularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

fig.add_subplot(122)

plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Regularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.show()