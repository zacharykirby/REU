from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


#set variables
img_size = 256
pixel_depth = 255.0
train_folders = []
test_folders = []
valid_folders = []
path = os.getcwd()

def create_folders():
    for x in range(4000):
        os.chdir(path + '\\' + str(x))
        train_folders.append(os.getcwd())
        #print(os.getcwd() + '- loaded')
    for x in range(4000,4500):
        os.chdir(path + '\\' + str(x))
        test_folders.append(os.getcwd())
        #print(os.getcwd() + '- loaded')
    for x in range(4500,5000):
        os.chdir(path + '\\' + str(x))
        valid_folders.append(os.getcwd())
        #print(os.getcwd() + '- loaded')

def load_anim(folder):
    frame_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(frame_files),img_size,img_size),
                         dtype=np.float32)
    print(folder)
    num_frames = 0
    for frame in frame_files:
        frame_file = os.path.join(folder,frame)
        try:
            #normalize data
            frame_data = (ndimage.imread(frame_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if frame_data.shape != (img_size,img_size):
                raise Exception('Unexpected image shape: %s' %
                                str(frame_data.shape))
            dataset[num_frames, :, :] = frame_data
            num_frames += 1
        except IOError as e:
            print('Could not read:',frame_file,':',e,'- skipping')
    dataset = dataset[0:num_frames, :, :]
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

create_folders()
##print(train_folders)
##print(test_folders)
##print(valid_folders)

#load animations
for folder in train_folders:
    load_anim(folder)

