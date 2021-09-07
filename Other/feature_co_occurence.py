from numpy.lib.type_check import common_type
import pandas as pd
import glob, os
import numpy as np
import scipy.io
import argparse

parser = argparse.ArgumentParser(
    description="Co-Occurence Matrix"
    )

# directory where segmented images and csv files are stored
parser.add_argument(
    "--imgs",
    required=True,
    type=str,
    help="an image path, or a directory name"
    )

# directory to save results
parser.add_argument(
    "--save_name",
    required=True,
    type=str,
    help="save directory name"
    )
    
args = parser.parse_args()    

# all_classes = []
# all_pixels_dict = {}
# all_pixels_dict_mean = {}
co_occurence_mat = np.zeros([150,150], dtype=int)
# os.chdir('/Volumes/Red_5TB/GLIMPSE_datasets/im2gps_seg/')
os.chdir(args.imgs)

for csvfile in glob.glob('*.csv'): # iterate through each file
    # print(csvfile)
# csvfile = '/Volumes/Red_5TB/GLIMPSE_datasets/im2gps_seg/Virginia_00012_710310620_0900169ce4_1266_94125865@N00.csv'
    df = pd.read_csv(csvfile)

    saved_column = np.array(df['Class Index'])
    saved_column.sort()

    # print('saved_column')
    # print(saved_column)
    for i in range(151):
        # if class i is present in saved_column:
        if i in saved_column:
            # print('class ' + str(i) + ' is present in file')

            for j in range(i, 151): # start at i
                # if class j is also present in saved_column:
                if j in saved_column:
                    # print('class ' + str(j) + ' is present in file')
                
                    # if they are different indices
                    if i != j: 
                        # print('co occurence of ' + str(i) + ' and ' + str(j) + ' in file' )
                        co_occurence_mat[i-1,j-1] = co_occurence_mat[i-1,j-1] + 1 # account for indices starting at 0 in python 
                        # symmetric
                        co_occurence_mat[j-1,i-1] = co_occurence_mat[j-1,i-1] + 1

    # print(co_occurence_mat[5-1,10-1])     

    # print(co_occurence_mat[5-1,13-1])     

    # print(co_occurence_mat[109-1, 127-1])     



# scipy.io.savemat('/Volumes/Red_5TB/My_Code/co_occurence.mat', {'co_occurence_mat': co_occurence_mat})
scipy.io.savemat(args.save_name, {'co_occurence_mat': co_occurence_mat})



            

