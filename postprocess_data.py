from __future__ import division
import os, scipy.io
from test_Sony import toimage
import tensorflow.compat.v1 as tf
from test_Sony import network
from d2s_numpy import depth_to_space
tf.disable_v2_behavior()
import numpy as np

output_filepath = "./qemu_output.data"
processed_output_filepath = "./qemu_output.png"
output_shape = (1, 2128, 2128, 12)
output_data = np.fromfile(output_filepath, dtype=np.float32).reshape(output_shape)
output_data = depth_to_space(output_data, 2)

# np.save("test_output_rawdata_nopost_{}".format(dtype), output_data[0, :, :, :])
output_data = np.minimum(np.maximum(output_data, 0), 1)  # Fits values between 0 and 1
# if "float" not in dtype or "bfloat" in dtype:
#    output_data = output_data / (1/np.max) # map to the maximum

output = output_data[0, :, :, :]
# np.save("test_output_rawdata_withpost_{}".format(dtype), output_data[0, :, : , :])

# print(max(output))

toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(processed_output_filepath)