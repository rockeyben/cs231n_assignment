import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test

def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_conv_out:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
    else:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])   

    train_mean = tf.cond(is_training, lambda: tf.assign(pop_mean,
                           pop_mean * decay + batch_mean * (1 - decay)), lambda: batch_mean)
    train_var = tf.cond(is_training, lambda: tf.assign(pop_var,
                          pop_var * decay + batch_var * (1 - decay)), lambda: batch_mean)
    with tf.control_dependencies([train_mean, train_var]):
        ans = tf.cond(is_training, lambda:tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001), lambda: tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001))
        return ans


def simple_model(X,y, is_training):
    # define our weights (e.g. init_two_layer_convnet)
    
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W2 = tf.get_variable("W2", shape=[1152, 1024])
    b2 = tf.get_variable("b2", shape=[1024])
    W3 = tf.get_variable("W3", shape=[1024, 10])
    b3 = tf.get_variable("b3", shape=[10])
    '''
    7x7 Convolutional Layer with 32 filters and stride of 1
    ReLU Activation Layer
    Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
    2x2 Max Pooling layer with a stride of 2
    Affine layer with 1024 output units
    ReLU Activation Layer
    Affine layer from 1024 input units to 10 outputs
    '''
    
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_bn = batch_norm(h1, is_training)
    # n * 13 * 13 * 32
    # print(h1_bn.shape)
    h2 = tf.layers.max_pooling2d(h1_bn, pool_size=(2, 2), strides=(2, 2), padding = 'VALID')
    # n * 6 * 6 * 32
    # print(h2.shape)
    h2_flat = tf.reshape(h2, [-1, 1152])
    a3 = tf.matmul(h2_flat, W2) + b2
    h3 = tf.nn.relu(a3)
    y_out = tf.matmul(h3,W3) + b3
    return y_out

