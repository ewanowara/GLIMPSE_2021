from numpy.lib.type_check import common_type
import pandas as pd
import glob, os
import numpy as np

all_classes = []
all_pixels_dict = {}
all_pixels_dict_mean = {}
os.chdir('/Volumes/Red_5TB/GLIMPSE_datasets/resources/images/im2gps/segmented_im2gps/')

for csvfile in glob.glob('*.csv'):
    # print(csvfile)
    # csvfile = '/Volumes/Red_5TB/GLIMPSE_datasets/resources/images/im2gps/segmented_im2gps/97344248_30a4521091_32_77325609@N00.csv'
    df = pd.read_csv(csvfile)
    saved_column = df['Class Index']  
    saved_percent = df['% of Image Pixels in Class']

    for i in range(saved_column.shape[0]):
        all_classes.append(saved_column[i]) 
        if saved_column[i] in all_pixels_dict: # if class already in dictionary, don't overwrite, but add the % pixels
            all_pixels_dict[saved_column[i]] = all_pixels_dict[saved_column[i]] + saved_percent[i]
        else:
            all_pixels_dict[saved_column[i]] = saved_percent[i]
            
all_classes.sort()
detected_classes = np.unique(all_classes)

common_classes = []
perc = []
for j in detected_classes:
    if 100*all_classes.count(j)/237 > 10:
        # print('percent of images ' + str(100*all_classes.count(j)/237) + ' in class ' + str(j))
        perc.append(100*all_classes.count(j)/237)
        common_classes.append(j)

        # average the saved % by number of each class
        if j in all_pixels_dict: 
            
            total_pix = all_pixels_dict[j]
            all_pixels_dict_mean[j] = total_pix/all_classes.count(j) # take a mean
        else:
            print('missing class: ' + str(j))


sort_idx = np.argsort(perc)

# print(np.array(common_classes)[sort_idx])        
# print(np.array(perc)[sort_idx])

print(all_pixels_dict_mean)