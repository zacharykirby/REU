#Zachary Kirby     <zkirby@live.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from six.moves import cPickle as pickle

#Load the animation ndarray from a pickled file
#return the ndarray
def load_animation(file):
    animation_file = open(file,'rb')
    animation = pickle.load(animation_file)
    animation_file.close()
    return animation

def cnn_model_fn(animation, labels, mode):
    #Input Layer
    #Reshape input to 4D tensor: [batch_size, width, height, channels]
    #Images 256x256, one color channel
    input_layer = tf.reshape(animation, [-1,256,256,1])

    #Convolution Layer
    #Computes 32 Features using 5x5 kernel with ReLU
    #Input Tensor: [frame_count,256,256,1]
    #Output Tensor: [frame_count,256,256,16]
    conv_layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=5,
        padding='SAME',
        activation=tf.nn.relu)

    #Pooling Layer
    #Max Pooling with 2x2 filter, stride of 2
    #Input Tensor Shape: [frame_count,256,256,16]
    #Output Tensor Shape: [frame_count,128,128,16]
    pool_layer = tf.layers.max_pooling2d(
        inputs=conv_layer,
        pool_size=2,
        strides=2)

    #Flatten tensor into batch of vectors
    #Input tensor: [frame_count, 128, 128, 16]
    #Output tensor: [frame_count, 128 * 128 * 16]
    pool_flat = tf.reshape(pool_layer, [-1,128*128*16])

    #Dense Layer
    #Create a densely connected layer of 1024 neurons
    #Input tensor: [frame_count, 128 * 128 * 16]
    #Output tensor: [frame_count, 1024]
    dense_layer = tf.layers.dense(
        inputs=pool_flat,
        units=1024,
        activation=tf.nn.relu)

    #Dropout op, 0.6 probability to keep element
    dropout = tf.layers.dropout(
        inputs=dense_layer,
        rate=0.4,
        training=mode == learn.ModeKeys.TRAIN)

    #Logits layer
    #Input tensor: [frame_count,1024]
    #Output tensor: [frame_count, 10]
    logits = tf.layers.dense(
        inputs=dropout,
        units=10)

    loss = None
    train_op = None

    #Calculate loss for TRAIN and EVAL
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    #Configure TRAIN op
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer='SGD')

    #Generate Predictions
    predictions = {
        'classes': tf.argmax(
            input=logits,
            axis=1),
        'probabilities': tf.nn.softmax(
            logits,
            name='softmax_tensor')
    }

    #return ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)

