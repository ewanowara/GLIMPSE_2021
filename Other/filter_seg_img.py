# from numpy.lib.type_check import common_type
import pandas as pd
import glob, os
import numpy as np
from PIL import Image
import scipy.io
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib

for root, dirs, files in os.walk("/Volumes/Red_5TB/GLIMPSE_datasets/yfcc25600_seg_images/"):
    for img_file in files:
        if img_file.endswith("3454782854.jpg"):
            if img_file.startswith(".") == 0:
                # print(os.path.join(root, img_file))
                img = np.array(Image.open(os.path.join(root, img_file)).convert("RGB"))
                img = img[:, int(img.shape[1]/2):int(img.shape[1]+1), :]

                csv_file = os.path.join(img_file.replace('.jpg', '.csv'))
                df = pd.read_csv(os.path.join(root, csv_file))
                class_idx = df['Class Index']  

                img_new = np.zeros((img.shape[0], img.shape[1], img.shape[2], class_idx.shape[0]), dtype=int)
                img_diff = np.zeros((img.shape[0], img.shape[1], img.shape[2], class_idx.shape[0]), dtype=int)
                # compute a difference tensor for each segmented class and original segmented image to find the closest class
                segment_file = scipy.io.loadmat('/Volumes/Red_5TB/My_Code/semantic_cagtegories.mat') # ('../../segmentation/CSAIL_semantic_segmentation/semantic_cagtegories.mat')
                color_categories = segment_file['color_categories']
                for c in range(class_idx.shape[0]):
                    RGB_idx = color_categories[class_idx[c]-1,:]
                    img_new[:,:,:,c] = img_new[:,:,:,c] + RGB_idx
                    img_diff[:,:,:,c] = abs(img_new[:,:,:,c] - img)
                
                # create new filtered segmented image by assigning each pixel to one of the pre-defined and detected segmentation classes
                img_new = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
                for row in range(img.shape[0]):
                    for col in range(img.shape[1]):
                        min_idx = np.argmin(sum(img_diff[row,col,:,:]))                       
                        img_new[row,col,:] = color_categories[class_idx[min_idx]-1, :]

                tmp_path = os.path.join(root, img_file)
                img_name = tmp_path.split('/Volumes/Red_5TB/GLIMPSE_datasets/yfcc25600_seg_images/')[-1]
                img_name2 = os.path.join(img_file.replace('.jpg', '.png'))
                save_folder = '/Volumes/Red_5TB/GLIMPSE_datasets/yfcc25600_seg_images_RERUN/'

                image_path = os.path.dirname(img_name)

                if os.path.exists(save_folder + image_path) == 0:
                    os.makedirs(save_folder + image_path)

                PIL_image1 = Image.fromarray(img_new) # TODO: issue when converting from np array to PIL - becomes blurry again
                PIL_image1.save(save_folder + image_path + '/' + img_name2)