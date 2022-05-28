from model_regularization_tutorial import get_regularized_model
import validation_sets_tutorial

print(validation_sets_tutorial.train_data)
model = get_regularized_model(1e-5, 0.3)
print(model.summary())