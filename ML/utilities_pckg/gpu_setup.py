import os
import tensorflow as tf
import keras.backend as K
import numpy as np
from tensorflow import set_random_seed

#np.random.seed(1234)
#set_random_seed(4321)


#   setup GPU before usage: this must be the first command before starting to set up networks otherwise it does not know
#   where to allocate memory
def gpu_setup(id_gpu, memory_percentage):
    #   only use GPU with index id_GPU
    #   id_GPU is a string like "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = id_gpu

    #   TensorFlow wizardry
    config = tf.ConfigProto()

    #   Don't pre_allocate memory; allocate as_needed
    config.gpu_options.allow_growth = True

    #   Only allow a total of half the GPU memory to be allocated
    #   memory_percentage is a float between 0 and 1
    config.gpu_options.per_process_gpu_memory_fraction = memory_percentage

    #   Create a session with the options specified above
    K.tensorflow_backend.set_session(tf.Session(config=config))
