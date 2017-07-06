#Zachary Kirby <zkirby@live.com>
#Code Source: Dr. Anthony S Maida

import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IM_SZ_LEN = 64
IM_SZ_WID = 64
BATCH_SZ = 1
NUM_UNROLLINGS = 3
LEARNING_RATE = 0.1
NUM_TRAINING_STEPS = 101

#define im4d
#http://learningtensorflow.com/lesson3/
file = 'image.jpg'
image = mpimg.imread(file)
print('Original Image Size: ',image.shape)

graph = tf.Graph()
with graph.as_default():

    x = tf.constant(image, dtype=tf.float32)
    x = tf.expand_dims(x,axis=0)
    print('Expanded Image Size: ',x.shape)

    U = tf.Variable(tf.truncated_normal([5,5,3,1],-0.1,0.1))
    W = tf.Variable(tf.truncated_normal([5,5,3,1],-0.1,0.1))
    B = tf.Variable(tf.ones([1,IM_SZ_LEN,IM_SZ_WID,1]))

    Ug = tf.Variable(tf.truncated_normal([5,5,3,1],-0.1,0.1))
    Wg = tf.Variable(tf.truncated_normal([5,5,3,1],-0.1,0.1))
    Bg = tf.Variable(tf.ones([1,IM_SZ_LEN,IM_SZ_WID,1]))

    Uf = tf.Variable(tf.truncated_normal([5,5,3,1],-0.1,0.1))
    Wf = tf.Variable(tf.truncated_normal([5,5,3,1],-0.1,0.1))
    Bf = tf.Variable(tf.ones([1,IM_SZ_LEN,IM_SZ_WID,1]))

    Uo = tf.Variable(tf.truncated_normal([5,5,3,1],-0.1,0.1))
    Wo = tf.Variable(tf.truncated_normal([5,5,3,1],-0.1,0.1))
    Bo = tf.Variable(tf.ones([1,IM_SZ_LEN,IM_SZ_WID,1]))

    initial_lstm_state = tf.Variable(tf.zeros([1,IM_SZ_LEN,IM_SZ_WID,3]))
    initial_lstm_output = tf.Variable(tf.zeros([1,IM_SZ_LEN,IM_SZ_WID,3]))

    #need input for error module (TODO LATER)
    def convLstmLayer(x,prev_s,prev_h):

        inp = tf.sigmoid(tf.nn.conv2d(x,U,[1,1,1,1],padding='SAME') +
                         tf.nn.conv2d(prev_h,W,[1,1,1,1],padding='SAME') + B)
        g_gate = tf.sigmoid(tf.nn.conv2d(x,Ug,[1,1,1,1],padding='SAME') +
                         tf.nn.conv2d(prev_h,Wg,[1,1,1,1],padding='SAME') + Bg)
        f_gate = tf.sigmoid(tf.nn.conv2d(x,Uf,[1,1,1,1],padding='SAME') +
                         tf.nn.conv2d(prev_h,Wf,[1,1,1,1],padding='SAME') + Bf)
        q_gate = tf.sigmoid(tf.nn.conv2d(x,Uo,[1,1,1,1],padding='SAME') +
                         tf.nn.conv2d(prev_h,Wo,[1,1,1,1],padding='SAME') + Bo)

        s = tf.multiply(f_gate,prev_s) + tf.multiply(g_gate,inp)
        h = tf.multiply(q_gate,tf.tanh(s)) #also try logsig or relu
        return s,h

    def errorModule(x,prediction):
        err1 = tf.nn.relu(x - prediction)
        err2 = tf.nn.relu(prediction - x)
        return tf.stack([err1,err2])

    #build lstm
    lstm_state = initial_lstm_state
    lstm_output = initial_lstm_output
    for i in range(NUM_UNROLLINGS):
        lstm_state, lstm_output = convLstmLayer(x,lstm_state,lstm_output)

    error_module_output = errorModule(x,lstm_output)

    loss = tf.reduce_sum(error_module_output)

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    
    for step in range(NUM_TRAINING_STEPS):
        i,l,predictions = session.run([optimizer,loss,lstm_output])

        print('Step: ', step)
        print('Loss: ',l)

        #IMAGE DISPLAY WIP
        if(step==100):
            print(predictions.shape)
            img = tf.squeeze(predictions)
            x = tf.Variable(img,dtype=tf.float32)
            plt.imshow(x)
            plt.show()
  
