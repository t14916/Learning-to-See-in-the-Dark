from __future__ import division
import os, scipy.io
from test_Sony import toimage
import tensorflow.compat.v1 as tf
from test_Sony import network
from d2s_numpy import depth_to_space
tf.disable_v2_behavior()
import numpy as np
import rawpy
import glob
#"8856"
#625


dtype = "int16"#"int8"#"float32"
#input_dir = './dataset/Sony/preprocessed_shorts/'
#input_dir = './dataset/Sony/preprocessed_shorts_padded/'
input_dir = './dataset/Sony/preprocessed_shorts_trunc'
checkpoint_in_dir = './checkpoint/Sony/'
checkpoint_out_dir = './checkpoint/QSony/'
#tflite_filepath = './checkpoint/SonyTFLite/tf2_sony_{}.tflite'.format(dtype)
#tflite_filepath = './checkpoint/SonyTFLite/tf2_sony_{}_pad.tflite'.format(dtype)
tflite_filepath = './checkpoint/SonyTFLite/tf2_sony_{}_trunc.tflite'.format(dtype)
#output_dir = './result_Sony/{}'.format(dtype)
output_dir = './result_Sony/{}_trunc'.format(dtype)

def representative_dataset():
    all_files = os.listdir(input_dir)
    print(len(all_files))
    for i in range(int(len(all_files)/2)):
        #arr = np.load("{}/{}".format(input_dir, all_files[i]))
        #arr = np.fromfile("{}/{}".format(input_dir, all_files[i]), dtype=np.single).reshape([1, 2128, 2128, 4])
        arr = np.fromfile("{}/{}".format(input_dir, all_files[i]), dtype=np.single).reshape([1, 1424, 1424, 4])
        yield [arr.astype(np.float32)]


def load_first_input():
    all_files = os.listdir(input_dir)
    print(all_files[0])
    return np.load("{}/{}".format(input_dir, all_files[0]))


def load_all_inputs(input_dir):
    all_files = os.listdir(input_dir)
    print(all_files[0])
    for file in all_files:
        #yield np.load("{}/{}".format(input_dir, file)), file
        #yield np.fromfile("{}/{}".format(input_dir, file), dtype=np.single).reshape([1, 2128, 2128, 4]), file
        yield np.fromfile("{}/{}".format(input_dir, file), dtype=np.single).reshape([1, 1424, 1424, 4]), file

def load_input(filename):
    print(filename)
    return np.load("{}/{}".format(input_dir, filename))


if __name__ == "__main__":
    sess = tf.Session()
    #in_image = tf.placeholder(tf.float32, [1, 1424, 2128, 4])
    #gt_image = tf.placeholder(tf.float32, [1, 2848, 4256, 3])
    #in_image = tf.placeholder(tf.float32, [1, 2128, 2128, 4])
    in_image = tf.placeholder(tf.float32, [1, 1424, 1424, 4])
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

    print("HELLO")
    converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [in_image], [out_image])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int16]
    #converter.target_spec.supported_types = [tf.int8]
    # Dont really need the following (causes some errors)
    #converter.experimental_new_converter = True
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    #CHANGE DATATYPE AT END OF THIS!!
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset
    tflite_quant_model = converter.convert()

    with open(tflite_filepath, 'wb') as f:
        f.write(tflite_quant_model)

    interpreter = tf.lite.Interpreter(tflite_filepath)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #https://www.tensorflow.org/lite/guide/inference
    for input_data, input_file in load_all_inputs(input_dir):
        print("Getting input")
        input_shape = input_details[0]['shape']
        #input_data = load_first_input()
        #input_data = load_input("10187_00_300_pinput.npy")
        print("Setting input")
        print(input_details[0]['index'])
        print(type(input_data[0]))
        print(input_details)
        print(input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        print("Beginning Run")
        interpreter.invoke()
        print("End Run")

        print("Begin Post Processing")
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = depth_to_space(output_data, 2)

        #np.save("test_output_rawdata_nopost_{}".format(dtype), output_data[0, :, :, :])
        output_data = np.minimum(np.maximum(output_data, 0), 1) # Fits values between 0 and 1
        #if "float" not in dtype or "bfloat" in dtype:
        #    output_data = output_data / (1/np.max) # map to the maximum

        output = output_data[0, :, :, :]
        #np.save("test_output_rawdata_withpost_{}".format(dtype), output_data[0, :, : , :])

        #print(max(output))

        toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save("{}/{}_{}.png".format(output_dir, input_file[:-4], dtype))
        print("Done")

