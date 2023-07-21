import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import ResNet50

#limit GPU memory Usage
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
         tf.config.experimental.set_virtual_device_configuration(
             gpus[0],
             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
         )
        except RuntimeError as e:
         print(e)

#path to dataset
current_directory = os.getcwd()
main_path = os.path.join(current_directory, 'Jupyter_Projects/Jupyter_Projects_server/train')
print(main_path)

#parameters
batch_size = 16
target_size = (224, 224)
num_classes = 1000

#load and preprocess dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    main_path,
    validation_split=0.2, 
    subset='training',
    seed=123,
    image_size=target_size,
    batch_size=batch_size
)

#configure dataset
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.shuffle(num_classes).prefetch(buffer_size=AUTOTUNE)

#create model (using ResNet50 model)
pretrained_model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
pretrained_model.trainable= False

model = Sequential([
    pretrained_model,
    tf.keras.Input(shape = (64,64,3)),
    tf.keras.layers.Dense(num_classes)
])

#compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#run model
model.fit(
  train_dataset,
  epochs=3
)