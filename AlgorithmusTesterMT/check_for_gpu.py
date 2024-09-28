import tensorflow as tf

physical_devices= tf.config.experimental.list_physical_devices('GPU')
print(len(physical_devices))


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))