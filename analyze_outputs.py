from __future__ import division
import os, scipy.io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import rawpy
import glob

def plot_and_save_variable(arr, save_name):
    plt.title("{} plot".format(save_name))
    plt.xlabel("values")
    plt.ylabel("count")
    plt.hist(arr)
    plt.savefig("{}".format(save_name))
    plt.clf()

def range(arr):
    return 6*np.std(arr)#np.amax(arr) - np.amin(arr)

plot_dir = "./plots/output/"
dtype = "float16"
nopost_output_file = "test_output_rawdata_nopost_{}.npy".format(dtype)
withpost_output_file = "test_output_rawdata_withpost_{}.npy".format(dtype)
arr1 = np.load(nopost_output_file)
arr2 = np.load(withpost_output_file)

index = 0
colors = ["R", "G", "B"]
rgb_arr1 = np.dsplit(arr1, arr1.shape[-1])
rgb_arr2 = np.dsplit(arr2, arr2.shape[-1])
print(arr1.shape)
for c in colors:
    carr1 = rgb_arr1[index]
    carr2 = rgb_arr2[index]
    print(carr1.shape)
    plot_and_save_variable(carr1.flatten(), "{}nopost/{}{}".format(plot_dir, c, nopost_output_file[:-3]))
    plot_and_save_variable(carr2.flatten(), "{}withpost/{}{}".format(plot_dir, c, withpost_output_file[:-3]))
    print("done plotting w/o readjustment")
    carr2 = carr2 * (1 / range(carr2))
    arr2[:, :, index] = np.reshape(carr2, newshape=carr2.shape[:-1])
    plot_and_save_variable(carr2.flatten(), "{}adj/{}{}".format(plot_dir, c, "test_output_adj_{}".format(dtype)))
    print("done plotting w/ readjustment")
    index = index + 1
    print(index)

output_data = np.minimum(np.maximum(arr2, 0), 1)
scipy.misc.toimage(output_data * 255, high=255, low=0, cmin=0, cmax=255).save("readjusted_output_{}.png".format(dtype))
