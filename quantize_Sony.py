from __future__ import division
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
"8856"
#625


dtype = "int8"
input_dir = './dataset/Sony/preprocessed_shorts/'
checkpoint_in_dir = './checkpoint/Sony/'
checkpoint_out_dir = './checkpiont/QSony/'
tflite_filepath = './checkpoint/SonyTFLite/sony_{}.tflite'.format(dtype)
def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out


def representative_dataset():
    all_files = os.listdir(input_dir)
    print(len(all_files))
    for i in range(int(len(all_files)/2)):
        arr = np.load("{}/{}".format(input_dir, all_files[i]))
        yield [arr.astype(np.float32)]


def load_first_input():
    all_files = os.listdir(input_dir)
    print(all_files[0])
    return np.load("{}/{}".format(input_dir, all_files[0]))

def load_input(filename):
    print(filename)
    return np.load("{}/{}".format(input_dir, filename))


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [1, 1424, 2128, 4])
gt_image = tf.placeholder(tf.float32, [1, 2848, 4256, 3])
out_image = network(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_in_dir)
print(ckpt)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

#output_data = sess.run(out_image, feed_dict={in_image: load_first_input()})
#print(representative_dataset())


converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [in_image], [out_image])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset
tflite_quant_model = converter.convert()

with open("v2_{}".format(tflite_filepath), 'wb') as f:
    f.write(tflite_quant_model)


#converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [in_image], [out_image])
#tflite_quant_model = converter.convert()

interpreter = tf.lite.Interpreter(tflite_filepath)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#https://www.tensorflow.org/lite/guide/inference
print("Getting input")
input_shape = input_details[0]['shape']
input_data = load_first_input()
#input_data = load_input("10187_00_300_pinput.npy")
print("Setting input")
print(input_details[0]['index'])
interpreter.set_tensor(input_details[0]['index'], input_data)

print("Beginning Run")
interpreter.invoke()
print("End Run")

print("Begin Post Processing")
output_data = interpreter.get_tensor(output_details[0]['index'])


np.save("test_output_rawdata_nopost_{}".format(dtype), output_data[0, :, : , :])
output_data = np.minimum(np.maximum(output_data, 0), 1) # Fits values between 0 and 1
#if "float" not in dtype or "bfloat" in dtype:
#    output_data = output_data / (1/np.max) # map to the maximum

output = output_data[0, :, :, :]
np.save("test_output_rawdata_withpost_{}".format(dtype), output_data[0, :, : , :])

#print(max(output))

scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save("test_output_{}.png".format(dtype))
print("Done")
