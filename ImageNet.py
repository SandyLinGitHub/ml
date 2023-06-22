import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import glob
from skimage import io
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import pathlib

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import ResNet50


current_directory = os.getcwd()
main_path = os.path.join(current_directory, 'Jupyter_Projects_server/ILSVRC2012_img_')
print(main_path)

batch_size = 32
target_size = (224, 224)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    main_path,
    #labels='inferred',
    #label_mode='categorical',  # Adjust based on your specific use case
    validation_split=0.2,  # Set the validation split ratio (e.g., 0.2 means 20% for validation)
    subset='training',
    seed=123,
    image_size=target_size,
    batch_size=batch_size
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

pretrained_model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
pretrained_model.trainable= False

model = Sequential([
    pretrained_model,
    tf.keras.Input(shape = (224,224,3)),
    tf.keras.layers.Dense(1000)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=5
model.fit(
  train_dataset,
  epochs=epochs
)
