from __future__ import division
import os, scipy.io
import numpy as np
import rawpy
import glob


input_dir = './dataset/Sony/short/'
output_dir = './dataset/Sony/preprocessed_shorts'
output_dir_cfile = './dataset/Sony/preprocessed_shorts_cfile'
output_dir_padded = './dataset/Sony/preprocessed_shorts_padded'
output_dir_trunc = './dataset/Sony/preprocessed_shorts_trunc'
gt_dir = './dataset/Sony/long/'

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

print(test_ids)
for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        #Begin input pre process
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)
        #End input pre process
        #print(type(input_full[0][0][0][0]))
        #input_full.tofile("{}/{}_00_{}_pinput.data".format(output_dir_cfile, test_id, ratio))
        height = input_full.shape[1]
        width = input_full.shape[2]
        pad_dims = abs(height-width) // 2
        #input_full = np.pad(input_full, ((0,0), (pad_dims,pad_dims), (0,0), (0,0)), "mean")
        input_full = input_full[:, :, pad_dims:height+pad_dims, :]#SHORTENED THIRD DIMENSION!!
        print(input_full.shape)
        #print(input_full.shape)
        #input_full.tofile("{}/{}_00_{}_pinput.data".format(output_dir_padded, test_id, ratio))
        input_full.tofile("{}/{}_00_{}_pinput.data".format(output_dir_trunc, test_id, ratio))
        #np.tofile("{}/{}_00_{}_pinput".format(output_dir, test_id, ratio), input_full)
