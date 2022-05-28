## Exploring models built in with Keras and usage

# Import and build Keras ResNet50 model
# Today we'll be using the ResNet50 model designed by a team at Microsoft Research, available through Keras applications. Please see the description on the Keras applications page for details. If you continue using it, please cite it properly! The paper it comes from is:
# Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition", 2015.
# This model takes a long time to download on the Coursera platform, so it is pre-downloaded in your workspace and saved in Keras HDF5 format. If you want to import it on your personal machine, use the following code:
# from tensorflow.keras.applications import ResNet50
# model = ResNet50(weights='imagenet')
# In this coding tutorial, you will instead load the model directly from disk.

## Just in case we want to download the ResNet50 model
from tensorflow.keras.applications import ResNet50 
model = ResNet50(weights = 'imagenet')

from tensorflow.keras.models import load_model

## The ResNet50 is a pretty big model, it starts with an input shape of (224, 224, 3)
## and final shape is 1000 neurons (1000 classes in the ImageNet challenge)

## Import and preprocess 3 sample images

from tensorflow.keras.preprocessing.image import load_img
lemon_img = load_img('data/lemon.jpg', target_size = (224, 224))
viaduct_img = load_img('data/viaduct_img.jpg', target_size = (224, 224))
water_tower_image = load_img('data/water_tower_image.jpg', target_size = (224, 224))

# Useful function: presents top 5 predictions and probabilities

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np 
import pandas as pd 

def get_top_5_predictions(img):
	x = img_to_array(img)[np.newaxis, ...]
	x = preprocess_input(x)
	preds = decode_predictions(model.predict(x), top = 5)
	top_preds = pd.DataFrame(columns = ['prediction', 'probability'],
		index = np.arange(5)+1)
	for i in range(5):
		top_preds.loc[i+1, 'prediction'] = preds[0][i][1]
		top_preds.loc[i+1, 'probability'] = preds[0][i][2]

	return top_preds 

## Get predictions for the lemon_img
get_top_5_predictions(lemon_img)

## The result shows 5 objects which resemble yellow circles 

