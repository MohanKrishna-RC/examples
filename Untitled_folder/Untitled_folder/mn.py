import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%%%matplotlib inline

function = lambda x: (x ** 2)-(5 *(x))+7

#Get 1000 evenly spaced numbers between -1 and 3 (arbitratil chosen to ensure steep curve)
x = np.linspace(-1,3,500)

#Plot the curve
plt.plot(x, function(x))
plt.show()

def deriv(x):
    x_der = 2*(x) - (10)
    return x_der

def step(x_new, x_prev, precision, l_r):
    x_list,y_list = [x_new],[function(x_new)]
    while abs(x_new - x_prev) > precision:
        x_prev = x_new
        d_r = -deriv(x_prev)
        x_new = x_prev + (l_r*d_r)
        x_list.append(x_new)
        y_list.append(function(x_new)) 
    print ("Local minimum occurs at: "+ str(x_new))
    print ("Number of steps: " + str(len(x_list)))
    plt.subplot(1,2,2)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,function(x), c="r")
    plt.title("Gradient descent")
    plt.show()
    plt.subplot(1,2,1)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,function(x), c="r")
    plt.xlim([1.0,2.1])
    plt.title("Zoomed in Gradient descent to Key Area")
    plt.show()


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev = 0.05))

def new _biases(length):
    return tf.Variable(tf.truncated_normal(shape = [length],stddev = 0.05))


def new_conv_layer(input,num_channels,filter_size,num_filters,use_pooling = True):
    shape = [filter_size,filter_size,num_filters,num_channels]
    weights = new_weights(shape = shape)
    biases = new_biases(shape = num_filters)

    layer = tf.nn.conv2d(input = input,weights = weights,strides = [1,1,1,1],padding = 'SAME')
    layer += biases

    if use_pooling:

        layer = tf.nn.max_pool(input = layer, ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'SAME')

    layer = tf.nn.relu(layer)

    return layer, weights

def new_fc_layer(input,num_inputs,num_outputs,use_relu = True):
    weights = new_weights(shape = [num_inputs,num_outputs])
    biases = new_biases(shape = [num_outputs])

    layer = tf.matmul(input,weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)
    
    return layer

def new_flatten_layer(layer):
    layer_shape = layer.get_shape()

    #layer_shape = [num_images,img_height,img_width,num_channels]

    num_features = layer_shape[1:4].num_elements()

    #num_features = inmg_height*img_width*num_channels

    layer_flat = tf.reshape(layer,[-1,num_features])

    return layer_flat , num_features


    
x = tf.placeholder(tf.float32, shape = [None,img_size_flat],name = 'x')
x_image = tf.reshape(x,[-1,img_size,img_size,num_channels])

y_true = tf.Variable(tf.float32,shape = [None,num_classes], name = 'y_true')

y_true_cls = tf.argmax(y_true , axis = 1)

