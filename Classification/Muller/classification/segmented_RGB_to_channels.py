import os
import sys
from PIL import Image
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

image_path = '/Volumes/Red_5TB/GLIMPSE_datasets/im2gps3k_seg/34489106_b6bb027a4c_23_49503208420@N01.jpg'
seg_csv = '/Volumes/Red_5TB/GLIMPSE_datasets/im2gps3k_seg/34489106_b6bb027a4c_23_49503208420@N01.csv'
# segment_class_idx = 5 # tree # Ewa TODO: pass this as a parameter to the test and inference scripts to change or loop over many classes

def segmented_RGB_to_channels(image_path, seg_csv):
    # input: a PIL Image with RGB concatenated with segmented parts, index of segmented class from 1 to 150
    # ouput: a PIL Image with RGB only and one segmented class masked out
    seg_img = np.array(Image.open(image_path)) #.convert("RGB"))
    df = pd.read_csv(seg_csv)
    # print(seg_csv)
    saved_column = df['Class Index']  

    # print(saved_column)
    
    # for segment_class_idx in saved_column: #= 3 
        # print(segment_class_idx) 
    # img_masked = img_RGB_seg[:, 0:int(img_RGB_seg.shape[1]/2) + 1, :]
    # seg_img = img_RGB_seg[:, int(img_RGB_seg.shape[1]/2):int(img_RGB_seg.shape[1]+1), :]

    # segment_class_idx = 17
    # # segment_file = scipy.io.loadmat('../../segmentation/CSAIL_semantic_segmentation/semantic_cagtegories.mat')
    # segment_file = scipy.io.loadmat('/Volumes/Red_5TB/My_Code/semantic_cagtegories.mat')
    # color_categories = segment_file['color_categories']

    # class_RGB = color_categories[segment_class_idx-1]
    # # print(class_RGB)

    # print('image')
    # print(seg_img[1:10,1,:].shape)
    # print(seg_img[1:10,1,:])

    # print(seg_img.shape)
    # seg_img1 = seg_img.reshape(786432, 3)

    u = np.unique(seg_img.reshape(-1, seg_img.shape[2]), axis=0) 
    print(u.shape)

    

    # seg_img1 = seg_img.reshape((seg_img[0]*seg_img[1], seg_img[2]))
    
    # print(seg_img1.shape)

    # unique_rows1 = np.unique(seg_img1, axis=1)
    # unique_rows1 = np.unique(unique_rows1, axis=0)

    # print(seg_img1[0:10,:])

    # print(unique_rows1)

    # unique_rows1 = np.unique(unique_rows1)
    # print(unique_rows1.shape)

    # unique_rows2 = np.unique(seg_img, axis=1)
    # print('second')
    # print(unique_rows2)

    # unique_rows3 = np.unique(seg_img, axis=1)
    # print('third')
    # print(unique_rows3)
    # indicesx, indicesy = np.where(np.all(np.array(seg_img) == tuple(class_RGB), axis=-1))

    # # plt.imshow(seg_img)
    # # plt.show()    

    # img_highlighted = np.multiply(0,seg_img) + 200# initialize with 0s
    # img_highlighted[indicesx, indicesy, :] = 255
    # img_highlighted = Image.fromarray(img_highlighted)

    # img_highlighted.save("img_highlighted" + str(segment_class_idx) + ".jpg") 

    # return img_highlighted

img_highlighted = segmented_RGB_to_channels(image_path, seg_csv)

