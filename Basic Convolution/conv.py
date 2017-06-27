#Zachary Kirby    <zkirby@live.com>

from __future__ import print_function
import matplotlib as plt
import numpy as np
from IPython.display import display, Image
from scipy import ndimage
import tensorflow as tf
from six.moves import cPickle as pickle

#Load the animation ndarray from a pickled file
#return the ndarray
def load_animation(file):
    animation_file = open(file,'rb')
    animation = pickle.load(animation_file)
    animation_file.close()
    return animation

###testing load method
##file = '0.pickle'
##print(load_animation(file))

#Build a convolution layer
def conv_layer(animation, shape_in, shape_out, name='conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5,5,size_in,size_out],
                                            stddev=0.1),name='W')
        b = tf.Varibale(tf.constant(0.1,shape=[size_out]), name='B')
        conv = tf.nn.conv2d(animation,w,strides=[1,1,1,1],padding='SAME')
        act = tf.nn.relu(conv + b)
        tf.summary.histogram('weights',w)
        tf.summary.histogram('biases',b)
        tf.summary.histogram('activations',act)
        return tf.nn.max_pool(act, ksize=[1,2,2,1],strides=[1,2,2,1],
                              padding='SAME')

#Build fully connected layer
def fc_layer(animation, shape_in, shape_out, name='fc'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([shape_in,shape_out],
                                            stddev=0.5), name='W')
        b = tf.Variable(tf.constant(0.1,shape=[shape_out]),name='B')
        act = tf.matmul(animation, w) + b
        tf.summary.histogram('weights',w)
        tf.summary.histogram('biases',b)
        tf.summary.histogram('activations',act)
        return act
    

#testing convolution on ndarray
file = '0.pickle'
animation = load_animation(file)
animation_shape = animation.shape
print(animation_shape)
print(fc_layer(animation,animation_shape,animation_shape))
