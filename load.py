from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
##from sklearn.linear_model import LogisticRegression
##from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


#set global variables
img_size = 256
pixel_depth = 255.0
train_folders = []
test_folders = []
valid_folders = []
path = os.getcwd()

#Define function to seperate the animation folders into training batches
#PARAMS: None
#RETURNS: None
def create_folders():
    
    #This will be our training batch
    for x in range(4000):
        os.chdir(path + '\\' + str(x))
        train_folders.append(os.getcwd())
        #print(os.getcwd() + '- loaded')

    #This will be our testing batch
    for x in range(4000,4500):
        os.chdir(path + '\\' + str(x))
        test_folders.append(os.getcwd())
        #print(os.getcwd() + '- loaded')

    #This will be our validation batch
    for x in range(4500,5000):
        os.chdir(path + '\\' + str(x))
        valid_folders.append(os.getcwd())
        #print(os.getcwd() + '- loaded')


#Define function to load animation folders into 3D arrays
#PARAMS: Folder of animation containing frames
#RETURNS: 3D ndarray with the animation data
def load_anim(folder):
    
    #frame_files is a list of all the frame file names
    frame_files = os.listdir(folder)
    #initialize dataset for loading and return
    dataset = np.ndarray(shape=(len(frame_files),img_size,img_size),
                         dtype=np.float32)
    #print(folder)

    #Loop to read in frame image data into an array, normalize it, then pass it
    #to the dataset
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

    #debugging statements
    dataset = dataset[0:num_frames, :, :]
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

#Define function to serialize the dataset created
#PARAMS: List of folders to be serialized
#RETURNS: list of the dataset names for testing
def maybe_pickle(folders,force=False):
    
    dataset_names = []

    #loop to check if the folder has been pickled or not
    for folder in folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            #pickle exists, skip it
            print('%s present - Skipping...' % set_filename)
        else:
            #pickle must be created, load the animation
            print('Pickling %s.' % set_filename)
            dataset = load_anim(folder)
            try:
                #dump data to pickle
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save to',set_filename,':',e)
    return dataset_names

#Create the training, test, and validation folder lists
create_folders()
##print(train_folders)
##print(test_folders)
##print(valid_folders)

#load animations
train_datasets = maybe_pickle(train_folders)
test_datasets = maybe_pickle(test_folders)
valid_datasets = maybe_pickle(valid_folders)

