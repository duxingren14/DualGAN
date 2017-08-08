import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

def batch_norm(x,  name="batch_norm"):
    eps = 1e-6
    with tf.variable_scope(name):
        nchannels = x.get_shape()[3]
        scale = tf.get_variable("scale", [nchannels], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        center = tf.get_variable("center", [nchannels], initializer=tf.constant_initializer(0.0, dtype = tf.float32))
        ave, dev = tf.nn.moments(x, axes=[1,2], keep_dims=True)
        inv_dev = tf.rsqrt(dev + eps)
        normalized = (x-ave)*inv_dev * scale + center
        return normalized

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
       
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def celoss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
       