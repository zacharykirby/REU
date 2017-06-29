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

#Build a convolution layer
#anim = ndarray of animation, shape=[frame_count,256,256]
#size_in = channels in (default to 1)
#size_out = channels out, also known as filter count
#Input Tensor: [frame_count,256,256,size_in]
#Output Tensor: [frame_count,256,256,size_out]
#...max pool output tensor: [frame_count,128,128,size_out]
def conv_layer(anim, size_in, size_out, name="conv"):
    
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out],
                                        stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(anim, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)

    #return a pooled output tensor (half size)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding="SAME")

#Dense Layer
#Input Tensor: [frame_count,128,128,size_in] (default size_in=128*128*64)
#Output Tensor: [frame_count,256,256,size_out]
def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.matmul(input, w) + b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


#testing convolution on ndarray
tf.reset_default_graph()
sess = tf.Session()

#placeholders? not quite sure what do
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
x_image = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', x_image, 3)
y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
learning_rate = 0.4

#load animation
file = '0.pickle'
animation = load_animation(file)

#build two convolution layers
#input tensor: [frame_count,256,256,1]
#output tensor: [frame_count,128,128,32]
conv1 = conv_layer(animation,1,32,'conv1')
#input tensor: [frame_count,128,128,32]
#output tensor: [frame_count,64,64,64]
conv_out = conv_layer(conv1,32,64,'conv2')

#flatten tensor into batch of vectors
#input tensor: [frame_count,64,64,64]
flatten = tf.reshape(conv_out,[-1,64*64*64])

#two dense layers, 1024 neurons
#input tensor: [-1,262144]
#output tensor: [-1,1024]
fc1 = fc_layer(flatten,64*64*64,1024,'fc1')
relu = tf.nn.relu(fc1)
embedding_input = relu
tf.summary.histogram('fc1/relu',relu)
embedding_size = 1024
#create a dense layer of 10 neurons
#input tensor: [-1,1024]
#output tensor: [-1,10]
logits = fc_layer(fc1,1024,10,'fc')

with tf.name_scope('xent'):
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,labels=y),name='xent')
    tf.summary.scalar('xent',xent)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

summ = tf.summary.merge_all()

embedding = tf.Variable(tf.zeros([1024,embedding_size]),name='test_embedding')
assignment = embedding.assign(embedding_input)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(os.getcwd())
writer.add_graph(sess.graph)

config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = embedding.name
embedding_config.sprite.image_path = SPRITES
embedding_config.metadata_path = LABELS
# Specify the width and height of a single thumbnail.
embedding_config.sprite.single_image_dim.extend([256, 256])
tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
