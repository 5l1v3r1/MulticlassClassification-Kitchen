from __future__ import absolute_import, division, print_function, unicode_literals

# for MacOS multithreading
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array

DATA_DIR = "ml-data/"

all_pictures = os.listdir(DATA_DIR)

if all_pictures.__contains__('DS_Store'):
    all_pictures.remove('.DS_Store') 

channels = 1
image_height = 340
image_width = 340

dataset = np.ndarray(shape=(len(all_pictures), image_height, image_width, channels),
                     dtype=np.float32)

i = 0
for pic_name in all_pictures:   
    image = load_img(DATA_DIR + pic_name,color_mode="grayscale") #color_mode="grayscale"
    image = img_to_array(image)
    image = image/255
    dataset[i] = image
    i += 1

labels = np.array((1,1,1,1,1,0,0,0,0),dtype='uint8')


# step 1 - initiate mode
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(340,340,1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

    
#keras.layers.Conv2D(64, (3, 3), input_shape=(340,340)),

# step 2 - compile model, set loss functionn 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# step 3 - train model
model.fit(dataset, labels, epochs=10)

x = model.predict(dataset, batch_size=9)

# test accuracy
test_loss, test_acc = model.evaluate(dataset,  labels, verbose=2)

print('\nTest accuracy:', test_acc)


def check_photo(x, i):
    plt.imshow(dataset[i].reshape(340,340))
    print(x[i][0])
    if x[i][0]>=1:
        print("It is a Dishplate")
    else:
        print("It is a glass")

check_photo(x, 2)
