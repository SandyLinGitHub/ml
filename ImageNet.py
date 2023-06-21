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

pretrained_model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
pretrained_model.trainable = False

pretrained_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
pretrained_model.fit(
  train_dataset,
  epochs=epochs
)



'''
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Set data_dir to a read-only storage of .tar files
# Set write_dir to a w/r storage
data_dir = 'datasets/imagenet/'
write_dir = 'psando/tf-imagenet-dirs'

# Construct a tf.data.Dataset
download_config = tfds.download.DownloadConfig(
                      extract_dir=os.path.join(write_dir, 'extracted'),
                      manual_dir=data_dir
                  )
download_and_prepare_kwargs = {
    'download_dir': os.path.join(write_dir, 'downloaded'),
    'download_config': download_config,
}
ds = tfds.load('imagenet2012_subset', 
               data_dir=os.path.join(write_dir, 'data'),         
               split='train', 
               shuffle_files=False, 
               download=True, 
               as_supervised=True,
               download_and_prepare_kwargs=download_and_prepare_kwargs)
'''




'''
dataset_dir = '/home/slin45/Jupyter_Projects/ILSVRC2012_img_train.tar'
temp_dir = '/home/slin45/Jupyter_Projects/temp'

download_config = tfds.download.DownloadConfig(
    extract_dir = os.path.join(temp_dir, 'extracted'),
    manual_dir = dataset_dir,
)

tfds.builder("imagenet2012").download_and_prepare(download_config=download_config)
'''