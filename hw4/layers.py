import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# def lrelu(x, leak=0.2, name="lrelu"):
#     return tf.maximum(x, leak*x)
#
# def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
#     shape = input_.get_shape().as_list()
#
#     with tf.variable_scope(scope or "Linear"):
#         matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
#         bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(bias_start))
#         if with_w:
#             return tf.matmul(input_, matrix) + bias, matrix, bias
#         else:
#             return tf.matmul(input_, matrix) + bias

def linear(input_, output_size, name ,scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("{}_M".format(name), [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("{}_b".format(name), [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
def rnncell(input, state, n_hidden=512, c=None, output=True, uniform_start=0.08, bias_start=0.0):
    '''
    input [batch_size, m]
    state [batch_size, n_hidden]
    c is the final state of encoder, or attention
    could be [batch_size, 2 * n_hidden] because it is the concatation of stats of two directions
    '''
    input_shape = input.get_shape().as_list()       # shape: [m * 1]
    state_shape = state.get_shape().as_list()       # shape: [n * 1]
    if c != None:
        c_shape = c.get_shape().as_list()[1]
    n = n_hidden
    m = input_shape[1]

    if c != None:
        C = tf.get_variable("C", [c_shape, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))
        Cz = tf.get_variable("Cz", [c_shape, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))
        Cr = tf.get_variable("Cr", [c_shape, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))

    Wz = tf.get_variable("Wz", [m, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))
    Uz = tf.get_variable("Uz", [n, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))
    bz = tf.get_variable("bz", [n], tf.float32, tf.constant_initializer(bias_start))
    if c != None:
        z = tf.sigmoid(tf.matmul(input, Wz) + tf.matmul(state, Uz) + tf.matmul(c, Cz)+ bz)
    else:
        z = tf.sigmoid(tf.matmul(input, Wz) + tf.matmul(state, Uz) + bz)

    Wr = tf.get_variable("Wr", [m, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))
    Ur = tf.get_variable("Ur", [n, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))
    br = tf.get_variable("br", [n], tf.float32, tf.constant_initializer(bias_start))
    if c != None:
        r = tf.sigmoid(tf.matmul(input, Wr) + tf.matmul(state, Ur) + tf.matmul(c, Cr) + br)
    else:
        r = tf.sigmoid(tf.matmul(input, Wr) + tf.matmul(state, Ur) + br)

    W = tf.get_variable("W", [m, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))
    U = tf.get_variable("U", [n, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))
    b = tf.get_variable("b", [n], tf.float32, tf.constant_initializer(bias_start))
    if c != None:
        h_ = tf.tanh(tf.matmul(input, W) + tf.matmul(tf.multiply(state, r), U) + tf.matmul(c, C) + b)
    else:
        h_ = tf.tanh(tf.matmul(input, W) + tf.matmul(tf.multiply(state, r), U) + b)

    h = tf.multiply(tf.ones([n]) - z, state) + tf.multiply(z, h_)

    if output:
        V = tf.get_variable("V", [n, n], tf.float32, tf.random_uniform_initializer(-uniform_start, uniform_start))
        V_b = tf.get_variable("V_b", [n], tf.float32, tf.constant_initializer(bias_start))
        y = tf.matmul(h, V) + V_b
        return h, y

    return h

def multi_rnncell(input, state, c=None, n_hidden=512, n_layers=4):
    '''
    input is [batch_size, m]
    state is [batch_size, n_layers, n_hidden], 0~(n_layers-1) is from the first layer to the last
    c is [batch_size, n_layers, (2*n_hidden)]
    '''
    # batch_size = input.get_shape().as_list()[0]
    # new_h = tf.zeros([batch_size, n_layers, n_hidden])
    h = []
    for i in range(n_layers):
        encoded = None if c is None else c[:,i,:]
        with tf.variable_scope("rnn_layer{}".format(i)) as scope:
            if i == 0:
                new_h, y = rnncell(input, state[:,i,:], c=encoded, n_hidden=n_hidden, output=True)     # With output
            else:
                new_h, y = rnncell(y, state[:,i,:], c=encoded, n_hidden=n_hidden, output=True)     # With output
            h += [new_h]
    # h is [layers, batch_size, n_hidden]
    return tf.transpose(tf.stack(h), perm=[1,0,2]), y
