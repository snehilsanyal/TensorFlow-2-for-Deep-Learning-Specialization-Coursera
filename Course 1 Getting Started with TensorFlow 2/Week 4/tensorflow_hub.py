# Import and build Tensorflow Hub MobileNet v1 model
# Today we'll be using Google's MobileNet v1 model, available on Tensorflow Hub. Please see the description on the Tensorflow Hub page for details on it's architecture, how it's trained, and the reference. If you continue using it, please cite it properly! The paper it comes from is:
# Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", 2017.
# This model takes a long time to download on the Coursera platform, so it is pre-downloaded in your workspace and saved in Tensorflow SavedModel format. If you want to import it on your personal machine, use the following code:
# module_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"
# model = Sequential([hub.KerasLayer(module_url)])
# model.build(input_shape=[None, 160, 160, 3])
# In this coding tutorial, you will instead load the model directly from disk

import tensorflow_hub as hub 
from tensorflow.keras.models import load_model, Sequential

module = load_model('models/Tensorflow_MobileNet_v1')
model = Sequential(hub.KerasLayer(module))
model.build(input_shape = [None, 160, 160, 3])
print(model.summary())
## Use MobileNet model to classify images 

from tensorflow.keras.preprocessing.image import img_to_array, load_img 
lemon_img = load_img('lemon_img.jpg', target_size = (160, 160))
viaduct_img = load_img('viaduct_img.jpg', target_size = (160, 160))
water_tower_img = load_img('water_tower_img.jpg', target_size = (160, 160))

## Read in categories text file
with open('data/imagenet_categories.txt') as txt_file:
	categories = txt_file.read().splitlines()

## Get Top 5 predictions
import numpy as np 
import pandas as pd 
def get_top_5_predictions(img):
	x = img_to_array(img)[np.newaxis, ...]/255.
	preds = model.predict(x)
	top_preds = pd.DataFrame(columns = ['prediction'], index = np.arange(5)+1)
	sorted_index = np.argsort(-preds[0])
	for i in range(5):
		ith_pred = categories[sorted_index[i]]
		top_preds.loc[i+1, 'prediction'] = ith_pred 
	return top_preds 
