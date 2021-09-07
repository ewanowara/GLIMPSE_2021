'''                
output a binary tensor with 150 channels, each corresponding to one of the segmentation classes      
'''
import pandas as pd
import glob, os
import numpy as np
from PIL import Image
import scipy.io
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image path, or a directory name"
    )
args = parser.parse_args()

# /Volumes/Red_5TB/GLIMPSE_datasets/yfcc25600_seg_images_RERUN/
start = time.process_time()

for root, dirs, files in os.walk(args.imgs):
    for img_file in files:
        if img_file.endswith(".png"):
            if img_file.startswith(".") == 0:
                if os.path.exists(os.path.join(root, img_file.replace('.png', '.mat'))) == 0: 
                    img = np.array(Image.open(os.path.join(root, img_file)).convert("RGB"))                
                    csv_file = os.path.join(img_file.replace('.png', '.csv'))
                    df = pd.read_csv(os.path.join(root, csv_file))
                    class_idx = df['Class Index']  

                    segment_file = scipy.io.loadmat('/Volumes/Red_5TB/My_Code/semantic_cagtegories.mat') # ('../../segmentation/CSAIL_semantic_segmentation/semantic_cagtegories.mat')
                    color_categories = segment_file['color_categories']

                    seg_channels = np.zeros((img.shape[0], img.shape[1], 150), dtype=bool)
                    for c in range(class_idx.shape[0]):
                        RGB_idx = color_categories[class_idx[c]-1,:]
                        mask = (img == RGB_idx).all(-1)
                        seg_channels[:,:, class_idx[c]-1] = mask
                    scipy.io.savemat(os.path.join(root, img_file.replace('.png', '.mat')), {'seg_channels': seg_channels})
print(time.process_time() - start)
                      


