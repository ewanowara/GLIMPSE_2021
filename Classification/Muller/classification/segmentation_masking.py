import os
import sys
from PIL import Image
import scipy.io
import numpy as np

# image_path = '/Volumes/Red_5TB/GLIMPSE_datasets/resources/images/im2gps/segmented_im2gps/195179890_dfc29db44a_70_73935513@N00.png'
# img_RGB_seg = np.array(Image.open(image_path).convert("RGB"))
# segment_class_idx = 5 # tree # Ewa TODO: pass this as a parameter to the test and inference scripts to change or loop over many classes

def mask_segmented_object(img_RGB_seg, segment_class_idx):
    # input: a PIL Image with RGB concatenated with segmented parts, index of segmented class from 1 to 150
    # ouput: a PIL Image with RGB only and one segmented class masked out

    img_masked = img_RGB_seg[:, 0:int(img_RGB_seg.shape[1]/2) + 1, :]
    seg_img = img_RGB_seg[:, int(img_RGB_seg.shape[1]/2):int(img_RGB_seg.shape[1]+1), :]

    segment_file = scipy.io.loadmat('../../segmentation/CSAIL_semantic_segmentation/semantic_cagtegories.mat')
    # segment_file = scipy.io.loadmat('/Users/ewanowara/Desktop/semantic_cagtegories.mat')
    color_categories = segment_file['color_categories']

    class_RGB = color_categories[segment_class_idx-1]
    indicesx, indicesy = np.where(np.all(np.array(seg_img) == tuple(class_RGB), axis=-1))

    img_masked[indicesx, indicesy, :] = 128 # half of 255 - middle grey value
    img_masked = Image.fromarray(img_masked)

    return img_masked

# img_masked = mask_segmented_object(image_path, segment_class_idx)
# img_masked.save("tmp/img_masked.jpg") # masked
