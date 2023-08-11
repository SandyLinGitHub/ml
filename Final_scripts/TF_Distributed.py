#
#  TensorFlow Distributed Training script
#
import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from models.ResNet import ResNet
from tensorflow.keras.preprocessing import image_dataset_from_directory

sess_start = time.time()        #session start time

# Enable AMP
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Set the number of threads for intra-op parallelism (operations within a graph)
intra_op_parallelism = 8  # Adjust this number as needed
tf.config.threading.set_intra_op_parallelism_threads(intra_op_parallelism)

# Set the number of threads for inter-op parallelism (operations between graphs)
inter_op_parallelism = 8  # Adjust this number as needed
tf.config.threading.set_inter_op_parallelism_threads(inter_op_parallelism)
#-------------------------------------------------------------------------------------------------------
# Distributed. Slightly different than non-distributed such that all listed GPU memory is limited to 10GB
#-------------------------------------------------------------------------------------------------------
#limit GPU memory Usage
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        # Limit GPU memory to 10GB
        gpu_memory_limit = 10*1024  # 10GB in MB
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)])
#-----------------------------------------------------------------------------------------------------
#path to dataset
current_directory = os.getcwd()
main_path = os.path.join(current_directory, 'Res_proj/Data_scripts/train')
print(main_path)

#parameters
batch_size = 64 * 2
target_size = (224, 224)
num_classes = 1000
ITERATIONS_PER_EPOCH = 8008
MAX_EPOCH = 1

#load and preprocess dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    main_path,
    validation_split=0.2, 
    subset='training',
    seed=123,
    image_size=target_size,
    batch_size=batch_size
)
#-----------------------------------------------------------------------------------------------------
# Distributed. Slightly different than non-distributed such that 
# ~ distribution strategy is defined
# ~ building and compiling the model both are done within the strategy scope to ensure that the model is built and compiled on both gpus. 
#-----------------------------------------------------------------------------------------------------
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())

with mirrored_strategy.scope():  
    #build model
    model = ResNet(50)        #change 50 to 18 if using ResNet18 model to train
    
    optimizer_name = tf.keras.optimizers.Adam()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer_name)
    
    #compile model
    model.compile(optimizer=optimizer,
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])
#-----------------------------------------------------------------------------------------------------
#training model
time_intervals = []
loss_intervals = []
accuracy_intervals = []

start = time.time()                  #training start time
history=model.fit(
    train_dataset,
    steps_per_epoch=ITERATIONS_PER_EPOCH,
    epochs=1
    )
curr_time = time.time()               #training end time
epoch_time = curr_time - start        #calculating training duration
time_intervals.append(epoch_time)
loss_intervals.append(history.history['loss'])
accuracy_intervals.append(history.history['accuracy']) 

results_df = pd.DataFrame({
    'Epochs': range(1,len(loss_intervals) + 1),
    'Loss': loss_intervals,
    'Accuracy': accuracy_intervals
})

results_df.to_csv('tr_50_DIS_Mirrored_TF.csv', index = False)

sess_end = time.time()                 #session end time
sess_dur = sess_end - sess_start       #calculating session duration
print("Session duration: ", sess_dur)

log_file = "sessLogDIS.txt"
with open(log_file, "a") as f:
    f.write(f"50_DIS_Mirrored_TF.py log")
    f.write(f"Session started at: {sess_start}\n")
    f.write(f"Session ended at: {sess_end}\n")
    f.write(f"Session duration: {sess_dur}\n")
    f.write(f"Session batch_size: {batch_size}\n")
    f.write(f"Session image_size: {target_size}\n")
    f.write(f"Session ITERATIONS_PER_EPOCHS: {ITERATIONS_PER_EPOCH}\n")
    f.write(f"Session completion time for ResNet18_ImageNet: {time_intervals}\n")
    f.write(f"LOSS: {loss_intervals}\n")
    f.write(f"ACCURACY: {accuracy_intervals}\n")
    f.write("50_Mirrored___________________________50_Mirrored\n")

