##
##  Nvidia-DALI data loader non-distributed
##

import nvidia.dali as dali
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf

import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from models.ResNet import ResNet

import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess_start = time.time()          #session start time

# Enable AMP
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

#Limit GPU Memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
         tf.config.experimental.set_virtual_device_configuration(
             gpus[0],
             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10*1024)]
         )
        except RuntimeError as e:
         print(e)
       
print("mem changed")

#dataset path using DALI 
os.environ['DALI_EXTRA_PATH'] = '/home/slin45/miniconda3/envs/tf/lib/python3.10/site-packages'
test_data_root = os.environ['DALI_EXTRA_PATH']
main_path = os.path.join(test_data_root, '/home/slin45/Res_proj/Data_scripts/train')
print(main_path)

#parameters
BATCH_SIZE = 64
IMAGE_SIZE = 224
ITERATIONS_PER_EPOCH = 16016
CHANNELS = 3
MAX_EPOCH = 1
target_size = (224, 224)
num_classes = 1000

#create DALI pipline, can't use TF image_dataset_from_directory
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

# Create dataset for cpu
with tf.device('/cpu:0'):
    img_set = dali_tf.DALIDataset(
        pipeline=pipeline,
        batch_size=BATCH_SIZE,
        output_shapes=shapes,
        output_dtypes=dtypes,
        device_id=0)

# Create dataset for gpu
with tf.device('/gpu:0'):
    img_set = dali_tf.DALIDataset(
        pipeline=img_pipeline(num_threads=16, device='gpu'),
        batch_size=BATCH_SIZE,
        output_shapes=shapes,
        output_dtypes=dtypes,
        device_id=0)
    
    model = ResNet(18)          #change 18 to 50 for ResNet50 model
    
    optimizer_name = tf.keras.optimizers.Adam()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_name)
    
    #compile model
    model.compile(optimizer=optimizer,
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

with tf.device('/gpu:0'):
    
    time_intervals = []
    loss_intervals = []
    accuracy_intervals = []
    
    start = time.time()                  #training start time
    history=model.fit(
        img_set,
        steps_per_epoch=ITERATIONS_PER_EPOCH,
        epochs=1
        )
    curr_time = time.time()              #training end time
    epoch_time = curr_time - start       #calculating training duration
    time_intervals.append(epoch_time)
    loss_intervals.append(history.history['loss'])
    accuracy_intervals.append(history.history['accuracy'])
    
    results_df = pd.DataFrame({
        'Epochs': range(1,len(loss_intervals) + 1),
        'Loss': loss_intervals,
        'Accuracy': accuracy_intervals
    })

    results_df.to_csv('tr_18_DALI.csv', index = False)

    sess_end = time.time()              #session end time
    sess_dur = sess_end - sess_start    #calculating session duration
    print("Session duration: ", sess_dur)

    log_file = "sessLogDALI.txt"
    with open(log_file, "a") as f:
        f.write(f"18_DALI.py log")
        f.write(f"Session started at: {sess_start}\n")
        f.write(f"Session ended at: {sess_end}\n")
        f.write(f"Session duration: {sess_dur}\n")
        f.write(f"Session batch_size: {BATCH_SIZE}\n")
        f.write(f"Session image_size: {target_size}\n")
        f.write(f"Session ITERATIONS_PER_EPOCHS: {ITERATIONS_PER_EPOCH}\n")
        f.write(f"Session completion time for ResNet18_ImageNet: {time_intervals}\n")
        f.write(f"LOSS: {loss_intervals}\n")
        f.write(f"ACCURACY: {accuracy_intervals}\n")
        f.write("18_DALI___________________________18_DALI\n")

  
