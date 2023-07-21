import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import ResNet50
import time
from tensorflow.compat.v1 import Session
import tensorflow.compat.v1 as tf_v1
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import argparse
import datetime

sess_start = time.time()
'''  
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''
#limit GPU memory Usage
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        # Limit GPU memory to 2GB
        gpu_memory_limit = 4*1024  # 4GB in MB
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)])

mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())

#path to dataset
current_directory = os.getcwd()
main_path = os.path.join(current_directory, 'Res_proj/train')
print(main_path)

parser = argparse.ArgumentParser()
parser.add_argument("--num_iter", type=int, default=10, help="Description of num_iter")
parser.add_argument("--loop_num", type=int, default=1, help="Description of loop_num")
args = parser.parse_args()
ITERATIONS_PER_EPOCH = args.num_iter
Loop_Num = args.loop_num

#parameters
batch_size = 16 * 2
target_size = (224, 224)
num_classes = 1000
#ITERATIONS_PER_EPOCH = 1000
CHANNELS = 3
MAX_EPOCH = 5

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
    tf.keras.Input(shape = (224,224,3)),
    tf.keras.layers.Dense(num_classes)
])

#compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#run model

log_dir = "50_TF_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = tf.keras.callbacks.History()

time_intervals = []
#loss_intervals = []
#accuracy_intervals = []

sum_time = 0
for i in range(MAX_EPOCH):
    start = time.time()
    hi=model.fit(
        train_dataset,
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
    #loss_intervals.append(history.history['loss'])
    #accuracy_intervals.append(history.history['accuracy'])

print("Epoch completion times for ResNet50_ImageNet: ", time_intervals)
print("Average completion time per epoch: ", sum_time/MAX_EPOCH)    

sess_end = time.time()
sess_dur = sess_end - sess_start
print("Session duration: ", sess_dur)

log_file = "session_log_50_TF_2.txt"
with open(log_file, "a") as f:
    f.write(f"Session loop num = {Loop_Num} for ResNet50 model using ImageNet dataset\n")
    f.write(f"Session started at: {sess_start}\n")
    f.write(f"Session ended at: {sess_end}\n")
    f.write(f"Session duration: {sess_dur}\n")
    f.write(f"Session ITERATIONS_PER_EPOCHS: {ITERATIONS_PER_EPOCH}\n")
    f.write(f"Session completion times for ResNet50_ImageNet: {time_intervals}\n")
    f.write(f"Session Average completion time/epoch: {sum_time/MAX_EPOCH}\n")
    #f.write(f"LOSS: {loss_intervals}\n")
    #f.write(f"ACCURACY: {accuracy_intervals}\n")
    f.write("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
