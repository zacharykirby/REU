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
            #normalize data to 0-1
            frame_data = (ndimage.imread(frame_file).astype(float)) / pixel_depth
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

def maybe_pickle(folders,force=False):
    dataset_names = []
    for folder in folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s present - Skipping...' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_anim(folder)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save to',set_filename,':',e)
    return dataset_names

create_folders()
##print(train_folders)
##print(test_folders)
##print(valid_folders)

#load animations
train_datasets = maybe_pickle(train_folders)
test_datasets = maybe_pickle(test_folders)
valid_datasets = maybe_pickle(valid_folders)

