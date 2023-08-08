import nvidia.dali as dali
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf

import os
import tensorflow as tf
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("/home/slin45/Res_proj/Data_scripts/ResNet.py")
from models.ResNet import ResNet

import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sess_start = time.time()

#limit GPU memory Usage
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        # Limit GPU memory to 2GB
        gpu_memory_limit = 10*1024  # 4GB in MB
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)])

mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())

print("mem changed")

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

shapes = (
    (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    (BATCH_SIZE))
dtypes = (
    tf.float32,
    tf.int32)

#mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())

with mirrored_strategy.scope():
    # Define and compile your model
    model = ResNet(18)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
def dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return dali_tf.DALIDataset(
                    pipeline=img_pipeline(device='gpu'),
                    batch_size=BATCH_SIZE,
                    output_shapes=shapes,
                    output_dtypes=dtypes,
                    device_id=device_id
    )

input_options = tf.distribute.InputOptions(
    experimental_place_dataset_on_device = True,
    experimental_fetch_to_device = False,
    experimental_replication_mode = tf.distribute.InputReplicationMode.PER_REPLICA)

train_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn, input_options)

# Perform training
time_intervals = []
loss_intervals = []
accuracy_intervals = []

sum_time = 0
for i in range(MAX_EPOCH):
    start = time.time()
    history = model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=ITERATIONS_PER_EPOCH
        #callbacks=[tensorboard_callback]
    )
    curr_time = time.time()
    epoch_time = curr_time - start
    sum_time = sum_time + epoch_time
    time_intervals.append(epoch_time)
    loss_intervals.append(history.history['loss'])
    accuracy_intervals.append(history.history['accuracy'])
    print(f"epoch={i}, time={curr_time - start: .3f}s")

print("Epoch completion times for ResNet18_ImageNet_DALI: ", time_intervals)
print("Average completion time per epoch: ", sum_time/MAX_EPOCH)

results_df = pd.DataFrame({
    'Epochs': range(1,len(loss_intervals) + 1),
    'Loss': loss_intervals,
    'Accuracy': accuracy_intervals
})

results_df.to_csv('tr_18_DALI_Mirrored.csv', index = False)
    
sess_end = time.time()
sess_dur = sess_end - sess_start
print("Session duration: ", sess_dur)
   
log_file = "sessLogDALI_Mirrored.txt"
with open(log_file, "a") as f:
    #f.write(f"Session loop num = {Loop_Num} for ResNet18 model using ImageNet dataset\n")
    f.write(f"18_DALI_Mirrored.py log")
    f.write(f"Session started at: {sess_start}\n")
    f.write(f"Session ended at: {sess_end}\n")
    f.write(f"Session duration: {sess_dur}\n")
    f.write(f"Session batch_size: {BATCH_SIZE}\n")
    f.write(f"Session image_size: {target_size}\n")
    f.write(f"Session ITERATIONS_PER_EPOCHS: {ITERATIONS_PER_EPOCH}\n")
    f.write(f"Session completion times for ResNet18_ImageNet: {time_intervals}\n")
    f.write(f"Session Average completion time/epoch: {sum_time/MAX_EPOCH}\n")
    f.write(f"LOSS: {loss_intervals}\n")
    f.write(f"ACCURACY: {accuracy_intervals}\n")
    f.write("18_DALI_Mirrored___________________________18_DALI_Mirrored\n")
