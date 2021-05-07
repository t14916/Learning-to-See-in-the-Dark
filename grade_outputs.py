import numpy as np
import os
import imageio
import glob

output_dir_fp32 = './result_Sony/float32_trunc'
output_dir_int8 = './result_Sony/int8_trunc'


def get_file_name(path):
    return path.split("/")[-1]


def get_all_image_filenames(input_dir):
    all_images = []
    for image_path in glob.glob("{}/*.png".format(input_dir)):
        """if "npy" not in image_path:
            print(image_path)
            print(get_file_name(image_path))
            os.rename(image_path, "./result_Sony/int8_trunc/{}".format(get_file_name(image_path)))"""
        #image = imageio.imread(image_path)
        all_images.append(image_path)
    return all_images


if __name__ == "__main__":
    int_files = get_all_image_filenames(output_dir_int8)
    int_files.sort()
    fp_files = get_all_image_filenames(output_dir_fp32)
    fp_files.sort()
    all_norms = []
    for int_file, fp_file in zip(int_files, fp_files):
        print(int_file)
        print(fp_file)
        int_image = imageio.imread(int_file).astype(np.int32)
        fp_image = imageio.imread(fp_file).astype(np.int32)
        normalized_int_image = int_image.astype(np.float32) / np.linalg.norm(int_image)
        normalized_fp_image = fp_image.astype(np.float32) / np.linalg.norm(fp_image)

        #print(int_image.shape)
        #print(fp_image.shape)
        #print(np.linalg.norm(fp_image))
        #print(np.linalg.norm(int_image))
        diff_image = normalized_fp_image - normalized_int_image
        #print(np.linalg.norm(diff_image)/np.linalg.norm(fp_image))
        all_norms.append(np.linalg.norm(diff_image))

    print(1 - max(all_norms))
    print(1 - min(all_norms))
    print(1 - sum(all_norms)/len(all_norms))