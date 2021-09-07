from numpy.lib.type_check import common_type
import pandas as pd
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io  

# os.chdir('/Volumes/Red_5TB/GeoEstimation-master/classification/results_on_occluded_Im2GPS/')
# for csvfile in glob.glob('*.csv'):
    # print(csvfile)

segment_class_idx_all = np.array([3,2,5,1,14,13,17,18,7,10,12,22,35,33,21,27,4,44])
performance_drop1 = []
performance_drop25 = []
performance_drop200 = []
performance_drop750 = []
performance_drop2500 = []

for segment_class_idx in segment_class_idx_all:           

    csvfile = '/Volumes/Red_5TB/GeoEstimation-master/classification/results_on_occluded_Im2GPS/test-segmented_im2gps' + str(segment_class_idx) + '.csv'
    df = pd.read_csv(csvfile)

    km1 = df['1']
    km25 = df['25']
    km200 = df['200']
    km750 = df['750']
    km2500 = df['2500']

    performance_drop1.append(0.15611814 - km1[2])
    performance_drop25.append(0.392405063 - km25[2])
    performance_drop200.append(0.489451468 - km200[2])
    performance_drop750.append(0.658227861 - km750[2])
    performance_drop2500.append(0.784810126 - km2500[2])

sort_idx1 = np.flip(np.argsort(performance_drop1))
sort_idx25 = np.flip(np.argsort(performance_drop25))
sort_idx200 = np.flip(np.argsort(performance_drop200))
sort_idx750 = np.flip(np.argsort(performance_drop750))
sort_idx2500 = np.flip(np.argsort(performance_drop2500))

# print('Performance drop for 1 km')
# print(np.array(segment_class_idx_all)[sort_idx1])
# print(np.array(performance_drop1)[sort_idx1])

# print('Performance drop for 25 km')
# print(np.array(segment_class_idx_all)[sort_idx25])
# print(np.array(performance_drop25)[sort_idx25])

# print('Performance drop for 200 km')
# print(np.array(segment_class_idx_all)[sort_idx200])
# print(np.array(performance_drop200)[sort_idx200])

# print('Performance drop for 750 km')
# print(np.array(segment_class_idx_all)[sort_idx750])
# print(np.array(performance_drop750)[sort_idx750])

# print('Performance drop for 2500 km')
# print(np.array(segment_class_idx_all)[sort_idx2500])
# print(np.array(performance_drop2500)[sort_idx2500])

semantic_cagtegories = scipy.io.loadmat('/Volumes/Red_5TB/My_Code/semantic_cagtegories.mat')
word_categories = semantic_cagtegories['word_categories_simplified']

# plt.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], 100*np.array(performance_drop1)[sort_idx1])
# plt.xlabel("Masked Objects")
# plt.ylabel("Performance Drop Over Baseline [%]")
# plt.title("1 km")
# plt.savefig('1km.svg')
# plt.savefig('1km.png')
# plt.close()

# plt.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], 100*np.array(performance_drop25)[sort_idx25])
# plt.xlabel("Masked Objects")
# plt.ylabel("Performance Drop Over Baseline [%]")
# plt.title("25 km")
# plt.savefig('25km.svg')
# plt.savefig('25km.png')
# plt.close()

# plt.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], 100*np.array(performance_drop200)[sort_idx200])
# plt.xlabel("Masked Objects")
# plt.ylabel("Performance Drop Over Baseline [%]")
# plt.title("200 km")
# plt.savefig('200km.svg')
# plt.savefig('200km.png')
# plt.close()

# plt.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], 100*np.array(performance_drop750)[sort_idx750])
# plt.xlabel("Masked Objects")
# plt.ylabel("Performance Drop Over Baseline [%]")
# plt.title("750 km")
# plt.savefig('750km.svg')
# plt.savefig('750km.png')
# plt.close()

# plt.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], 100*np.array(performance_drop2500)[sort_idx2500])
# plt.xlabel("Masked Objects")
# plt.ylabel("Performance Drop Over Baseline [%]")
# plt.title("2500 km")
# plt.savefig('2500km.svg')
# plt.savefig('2500km.png')
# plt.close()

class_names_1 = word_categories[np.array(segment_class_idx_all)[sort_idx1]-1]
class_names_25 = word_categories[np.array(segment_class_idx_all)[sort_idx25]-1]
class_names_200 = word_categories[np.array(segment_class_idx_all)[sort_idx200]-1]
class_names_750 = word_categories[np.array(segment_class_idx_all)[sort_idx750]-1]
class_names_2500 = word_categories[np.array(segment_class_idx_all)[sort_idx2500]-1]


print('class_names_1')
print(class_names_1)
print('class_names_25')
print(class_names_25)
print('class_names_200')
print(class_names_200)
print('class_names_750')
print(class_names_750)
print('class_names_2500')
print(class_names_2500)