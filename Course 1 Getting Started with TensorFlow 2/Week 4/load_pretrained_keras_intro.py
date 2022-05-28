## We can use Keras pre-trained model from keras.io to use as feature extractor

from tensorflow.keras.applications.resnet50 import ResNet50 
# Model architecture and weights will be downloaded and stored in the system
# The model will be loaded with weights that have been learned from training on the ImageNet dataset
model = ResNet50(weights = 'imagenet')
# If we want the weights to be randomly reinitialized, we can set the weights argument to None
model = ResNet50(weights = None)
# Another argument is include_top, which is True by default, and then downloads the complete classifier
# model, if set to False, the FC layer at the top of the network is not loaded, which gives us a headless
# model, useful for transfer learning and other applications.
model = ResNet50(weights = None, include_top = False)

# Let us predict the class of image using the downloaded ResNet50 model

from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np 

model = ResNet50(weights = 'imagenet', include_top = True)

img_input = image.load_img('test.jpg', target_axis = (224, 224))
img_input = image.img_to_array(img_input)
## The preprcessing is done using a module from tf.keras.applications.resnet50
img_input = preprocess_input(img_input[np.newaxis, ...])

preds = model.predict(img_input)
decoded_predictions = 	decode_predictions(preds, top = 3)[0]

# decoded_predictions returns a List of (class, description, probability)
# top = 3, means we are retrieving the top 3 model predictions