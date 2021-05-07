from __future__ import division
import os, scipy.io
from test_Sony import toimage
import tensorflow.compat.v1 as tf
import tf_slim as slim
tf.disable_v2_behavior()
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import rawpy
import glob
"8856"
#625


dtype = "float16"
input_dir = './dataset/Sony/preprocessed_shorts/'
checkpoint_in_dir = './checkpoint/Sony/'
checkpoint_out_dir = './checkpiont/QSony/'
tflite_filepath = './checkpoint/SonyTFLite/sony_{}.tflite'.format(dtype)
plot_dir = './plots/'
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


def replace_slash_with_dash(string):
    return string.replace("/", "-")


def plot_and_save_variable(var, save_dir):
    var_name = replace_slash_with_dash(var.name)
    plt.title("{} plot".format(var_name))
    plt.xlabel("values")
    plt.ylabel("count")
    plt.hist(var.eval(sess).flatten())
    plt.savefig("{}/{}".format(save_dir, var_name))
    plt.clf()

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

all_variables = tf.compat.v1.trainable_variables()
print(all_variables)
print(len(all_variables))

weights = []
biases = []
other = []
for var in all_variables:
    if "weights" in var.name:
        weights.append(var)
    elif "biases" in var.name:
        biases.append(var)
    else:
        other.append(var)

print(other)
print(len(weights))
print(len(biases))


for weight in weights:
    plot_and_save_variable(weight, "{}weights".format(plot_dir))

for bias in biases:
    plot_and_save_variable(bias, "{}biases".format(plot_dir))



