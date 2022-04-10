import numpy as np
from matplotlib.pyplot import imread
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2
from keras.models import load_model
# load model
model = load_model('model/model.h5')
img_path = 'cat.jpg'
#read image
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
my_image = imread(img_path)
preds=model.predict(x)

# "0": "cat", "1": "dog"
print( preds )    # probabilities for being in each of the 2 classes    
