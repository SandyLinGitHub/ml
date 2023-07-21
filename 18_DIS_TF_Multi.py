##
##
## ResNet18 model using ImageNet script runnable. 
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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

########################
#   enable uvm
########################

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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
#os.environ['DALI_EXTRA_PATH'] = '/.local/lib/python3.10/site-packages/'
#test_data_root = os.environ['DALI_EXTRA_PATH']
curr_dir = os.getcwd()
main_path = os.path.join(curr_dir, 'Git_ml/train')
print(main_path)

target_size = (224,224)
batch_size = 8
num_threads = 4
device_id = 0
num_iterations = 10

IMAGE_SIZE = (224, 224)
PER_WORKER_BATCH_SIZE = 8
NUM_WORKERS = len(tf_config['cluster']['worker'])
GLOBAL_BATCH_SIZE = PER_WORKER_BATCH_SIZE * NUM_WORKERS
EPOCHS = 1


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

#############################
#   create ResNet18 model
#############################

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

#input_shape = (224, 224, 3)
#num_classes = 1000  # Number of output classes
#model = ResNet18(input_shape, num_classes)

with strategy.scope(): 
    input_shape = (224, 224, 3)
    num_classes = 1000  # Number of output classes
    model = ResNet18(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

history = model.fit(
    train_dataset, 
    epochs = EPOCHS, 
    verbose = 1,
)
