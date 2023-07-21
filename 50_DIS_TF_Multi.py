##
##
## ResNet50 model using ImageNet dataset script runs fine.
## Did not run into memory allocation issues where there is the need to enable uvm.
## GPU usage on master node is steady around 1800-1910MiB
##
##
import os
import json
import tensorflow as tf
from tensorflow import _keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
import pathlib

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)
########################
#   configure gpu
########################

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

########################
#   create cluster
########################

tf_config = {
    'cluster': {
        'worker': ['128.230.211.197:2222','128.230.211.96:2222']
    },
    'task': {'type': 'worker', 'index': 0}
}
os.environ.pop('TF_CONFIG', None)
os.environ['TF_CONFIG'] = json.dumps(tf_config)

########################
#   create stratgey
########################

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options)

########################
#   preparing data
########################

curr_dir = os.getcwd()
main_path = os.path.join(curr_dir, 'Git_ml/train')
print(main_path)

target_size = (224,224)
batch_size = 8
num_threads = 4
device_id = 0
num_iterations = 10

IMAGE_SIZE = 224
PER_WORKER_BATCH_SIZE = 8
NUM_WORKERS = len(tf_config['cluster']['worker'])
GLOBAL_BATCH_SIZE = PER_WORKER_BATCH_SIZE * NUM_WORKERS
EPOCHS = 1
CHANNELS = 3


train_dataset = image_dataset_from_directory(
    main_path,
    validation_split=0.2,
    subset = "training",
    seed = 123,
    image_size = target_size,
    batch_size = batch_size,
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)

with strategy.scope(): 
    model = tf.keras.models.Sequential([
      tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet',input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
      tf.keras.layers.Dense(1000)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

history = model.fit(
    train_dataset, 
    epochs = EPOCHS, 
    verbose = 1,
)
