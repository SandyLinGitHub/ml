##
## RestNet18 model is runnable
## just by viewing the iterations progress, it is visably knowticeable to be quicker than the ResNet50 model
##

import nvidia.dali as dali
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf

import os
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_models as tfm
from official.vision.serving import export_saved_model_lib
import official.core.train_lib
from tensorflow.compat.v1 import Session

import time
import argparse
import datetime

import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sess_start = time.time()

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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
         
print("mem changed")

#os.environ['DALI_EXTRA_PATH'] = '/root/.cache/pip/wheels/5c/13/5f/51ffa68fc8ea8ac85574e82f6dc0891cec92078096b7f90ec5'
os.environ['DALI_EXTRA_PATH'] = '/.local/lib/python3.10/site-packages/'
test_data_root = os.environ['DALI_EXTRA_PATH']
#data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/MNIST/tr
#joins the DALI environment with the relative path to the dataset
main_path = os.path.join(test_data_root, '/home/slin45/Git_ml/train')
#main_path2 = 'Git_ml/train' 
print(main_path)

parser = argparse.ArgumentParser()
parser.add_argument("--num_iter", type=int, default=10, help="Description of num_iter")
parser.add_argument("--loop_num", type=int, default=1, help="Description of loop_num")
args = parser.parse_args()
ITERATIONS_PER_EPOCH = args.num_iter
Loop_Num = args.loop_num

#parameters
BATCH_SIZE = 16
IMAGE_SIZE = 224
#ITERATIONS_PER_EPOCH = 100
CHANNELS = 3
MAX_EPOCH = 5

#create own pipline, can't use image_dataset_from_directory
@pipeline_def(device_id=0, batch_size=BATCH_SIZE)
def img_pipeline(device):
  jpegs, labels = fn.readers.file(file_root=main_path, random_shuffle=True)
  images = fn.decoders.image(
    jpegs, device='mixed' if device == 'gpu' else 'cpu', output_type=types.RGB)
  images = fn.resize(images, resize_x = 224, resize_y = 224)
  images = fn.crop_mirror_normalize(
    images, device=device, dtype=types.FLOAT, std=[255.], output_layout="HWC")
  if device == 'gpu':
        labels = labels.gpu()

  return images, labels

# Create pipeline
pipeline = img_pipeline(device='cpu')

# Define shapes and types of the outputs
shapes = (
    (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    (BATCH_SIZE))
dtypes = (
    tf.float32,
    tf.int32)

# Create dataset
with tf.device('/cpu:0'):
    img_set = dali_tf.DALIDataset(
        pipeline=pipeline,
        batch_size=BATCH_SIZE,
        output_shapes=shapes,
        output_dtypes=dtypes,
        device_id=0)
    
def resnet_block(inputs, filters, strides=1, downsample=False):
    identity = inputs
    shortcut = tf.keras.layers.Conv2D(filters, 1, strides=strides, padding='same')(inputs)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if downsample:
        identity = tf.keras.layers.Conv2D(filters, 1, strides=strides, padding='same')(inputs)
        identity = tf.keras.layers.BatchNormalization()(identity)

    x = tf.keras.layers.Add()([x, identity])
    x = tf.keras.layers.ReLU()(x)

    return x

def ResNet18(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, strides=2, downsample=True)
    x = resnet_block(x, 128)
    x = resnet_block(x, 256, strides=2, downsample=True)
    x = resnet_block(x, 256)
    x = resnet_block(x, 512, strides=2, downsample=True)
    x = resnet_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model

# Create ResNet18 model
input_shape = (224, 224, 3)
num_classes = 1000  # Number of output classes
model = ResNet18(input_shape, num_classes)

# Create dataset
with tf.device('/gpu:0'):
    img_set = dali_tf.DALIDataset(
        pipeline=img_pipeline(device='gpu'),
        batch_size=BATCH_SIZE,
        output_shapes=shapes,
        output_dtypes=dtypes,
        device_id=0)
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

with tf.device('/gpu:0'):
    
    log_dir = "18_DALI_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = tf.keras.callbacks.History()
    
    time_intervals = []
    sum_time = 0
    for i in range(MAX_EPOCH):
        start = time.time()
        hi=model.fit(
        img_set,
        epochs=1,
        steps_per_epoch=ITERATIONS_PER_EPOCH,
        callbacks=[tensorboard_callback]
        )
        print(type(hi))
        curr_time = time.time()
        epoch_time = curr_time - start
        sum_time = sum_time + epoch_time
        time_intervals.append(epoch_time)
        print(f"epoch={i}, time={curr_time - start: .3f}s")

    print("Epoch completion times for ResNet18_ImageNet_DALI: ", time_intervals)
    print("Average completion time per epoch: ", sum_time/MAX_EPOCH)    

    sess_end = time.time()
    sess_dur = sess_end - sess_start
    print("Session duration: ", sess_dur)
    
    log_file = "session_log_18_DALI.txt"
    with open(log_file, "a") as f:
        f.write(f"Session loop num = {Loop_Num} for ResNet18 model using ImageNet dataset\n")
        f.write(f"Session started at: {sess_start}\n")
        f.write(f"Session ended at: {sess_end}\n")
        f.write(f"Session duration: {sess_dur}\n")
        f.write(f"Session ITERATIONS_PER_EPOCHS: {ITERATIONS_PER_EPOCH}\n")
        f.write(f"Session completion times for ResNet18_ImageNet: {time_intervals}\n")
        f.write(f"Session Average completion time/epoch: {sum_time/MAX_EPOCH}\n")
        #f.write(f"LOSS: {loss_intervals}\n")
        #f.write(f"ACCURACY: {accuracy_intervals}\n")
        f.write("#################################################\n")
        f.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
