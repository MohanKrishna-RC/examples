import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage
from random import shuffle
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/mohan/Traffic_signals"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)

print(type(images))

# # Note that the images and labels variables are lists,
# # so you might need to use np.array() to convert the variables to an array in your own workspace. 

# #Convert the lists to arrays.
images = np.array(images)
labels = np.array(labels)
print(images.ndim)

print(images.size)

print(images.flags)     # np.flags --> Information about the memory layout of the array.

print(images.itemsize)  # np.itemsize --> Length of one array element in bytes.

print(images.nbytes)    # np.nbytes --> Total bytes consumed by the elements of the array.


print(labels.ndim)
print(labels.size)
num_classes = len(set(labels))  #To get the number of labels in our dataset


plt.hist(labels,62)
# plt.show()

# Determine the (random) indexes of the images that you want to see

"""
One-hot encoding is compulsory here
"""
# Integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

# Binary encode

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
f_labels = onehot_encoder.fit_transform(integer_encoded)

print(f_labels.shape)
print(f_labels)

traffic_signs = [300, 2250, 3650, 4000]

# # Fill out the subplots with the random images that you defined 
def plot_images(traffic_signs):
    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(images[traffic_signs[i]])
        plt.subplots_adjust(wspace=0.5)
        plt.show()
        print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                    images[traffic_signs[i]].min(), 
                                                    images[traffic_signs[i]].max()))

# plot_images(traffic_signs)
# Plotting an overview of all the 62 classes and one image that belongs to each class:

labels = labels.tolist()

# # Get the unique labels 
unique_labels = set(labels)

# # Initialize the figure
plt.figure(figsize=(15, 15))

# # Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)
    
# Show the plot
# plt.show()

# Import the `transform` module from `skimage`
from skimage import transform 

# Rescale the images in the `images` array
images32 = [transform.resize(image, (32, 32)) for image in images]

images32 = np.array(images32)
print(images32.shape)

r_idxes = np.arange(images32.shape[0])
np.random.shuffle(r_idxes)
print(r_idxes)

s_images32 = images32[r_idxes]
print(s_images32.shape)
s_labels = f_labels[r_idxes]
print(s_labels.shape)
#While converting to grayscale images we have to convert the list images28 to array again,
# as the function rgb2gray will take array as an argument

# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray

# Convert `images28` to an array
# images28 = np.array(images28)
# print(images28.shape)

# Convert `images28` to grayscale
# images32 = rgb2gray(images32)
# print(images28.shape)
images32 = np.reshape(images32, [-1,32,32,1])
print(images32.shape)
# print(x)
num_classes = 62
# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 32,32,1])
y = tf.placeholder(dtype = tf.float32, shape = [None,num_classes])
keep_prob = tf.placeholder(tf.float32)
# x = tf.reshape(x_s, [-1,28,28,1])
# ddtype = 
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')
# with tf.name_scope('Wx_b') as scope:
weights = {
    
    # 'w1': tf.get_variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    'w1': tf.get_variable("w1", shape=[3,3,1,64],initializer=tf.contrib.layers.xavier_initializer()),
    # 'w2': tf.Variable(tf.random_normal([3, 3, 32, 64],stddev=0.1)),

    # 'w3': tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.1)),
    
    # 'wf1': tf.get_variable(tf.random_normal([32*32*64, 1024],stddev=0.1)),
    'wf1': tf.get_variable("wf1", shape=(32*32*64,1024),initializer=tf.contrib.layers.xavier_initializer()),
    # 'wf2': tf.Variable(tf.random_normal([1024,256],stddev=0.1)),
    # 'wf3': tf.Variable(tf.random_normal([256,128],stddev=0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    # 'out': tf.get_variable(tf.random_normal([1024, num_classes],stddev=0.1))
    'out': tf.get_variable("out  8u", shape=(1024, num_classes),initializer=tf.contrib.layers.xavier_initializer())

}

biases = {
    'b1': tf.get_variable(tf.constant(0.0, shape=[64])),
    # 'b2': tf.Variable(tf.constant(0.0, shape=[64])),
    # 'b3': tf.Variable(tf.constant(0.0, shape=[64])),
    'bf1': tf.get_variable(tf.constant(0.0, shape=[1024])),
    # 'bf2': tf.Variable(tf.constant(0.0, shape=[256])),
    # 'bf3': tf.Variable(tf.constant(0.0, shape=[128])),
    'out': tf.get_variable(tf.constant(0.0, shape=[num_classes]))
}


# # Creating model


def conv_net(x, weights, biases, dropout):
    
    # Convolution Layer
    conv1 = conv2d(x, weights['w1'], biases['b1'])
    # Max Pooling (down-sampling)
    # conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # conv2 = conv2d(conv1, weights['w2'], biases['b2'])
    # Max Pooling (down-sampling)
    # conv2 = maxpool2d(conv2, k=2)
    
    # Convolution Layer
    # conv3 = conv2d(conv2, weights['w3'], biases['b3'])
    # Max Pooling (down-sampling)
    # conv3 = maxpool2d(conv3, k=2)
    # conv1 = tf.nn.dropout(conv1, dropout)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(x, [-1, weights['wf1'].get_shape().as_list()[0]])
    print(fc1)
    fc1 = tf.add(tf.matmul(fc1, weights['wf1']), biases['bf1'])
    print(fc1)
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = tf.add(tf.matmul(fc1, weights['wf2']), biases['bf2'])
    # fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    # fc2 = tf.nn.dropout(fc2, dropout)
    
    # fc3 = tf.add(tf.matmul(fc2, weights['wf3']), biases['bf3'])
    # fc3 = tf.nn.relu(fc3)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

learning_rate = 0.01

# keep_prob = tf.placeholder()
logits = conv_net(x, weights, biases, keep_prob)
print(logits)
prediction = tf.nn.softmax(logits)
# More name scopes will clean up graph representation
with tf.name_scope("loss") as scope:
    # Minimize error using cross entropy
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logits))

    tf.summary.scalar("loss",loss_op)

with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)    # Create a summary to monitor the cost function

with tf.name_scope("Accuracy") as scope:
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("Accuracy", accuracy)

# Initializing the variables
init = tf.global_variables_initializer()
# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

save_file = './saved_models1/tensorflow_model_0.ckpt' 
saver = tf.train.Saver()


# print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss_op)
print("predicted_labels: ", correct_pred)

# for i in range(100):
#     print( i )
#     image_batch, label_batch = tf.train.batch([images32, labels],batch_size=20)
tf.set_random_seed(1234)
batch_size = 32
import time
with tf.Session() as sess:
    start_time = time.time()
    sess.run(init)
    writer = tf.summary.FileWriter('./logs',graph = sess.graph)
    for i in range(12):
        print('EPOCH', i)
        for step in range(int(len(images32)/batch_size)):
            b_img = images32[step*32:(step+1)*32,:,:,:]
            b_lab = f_labels[step*32:(step+1)*32]
            _, loss_val,Accuracy = sess.run([train_op, loss_op, accuracy], feed_dict={x: b_img, y: b_lab,keep_prob:0.5})
            # print("Accuracy: ", Accuracy)
        if i % 4 == 0:
            print("Loss:{} ", loss_val)
            print("Accuracy: ", Accuracy)
        print('DONE WITH EPOCH')
    end_time = time.time()
    diff_time = end_time - start_time
    print("time_taken : ",diff_time)

# #Prediction

# import random

# # Pick 10 random images
# sample_indexes = random.sample(range(len(images32)), 10)
# sample_images = [images32[i] for i in sample_indexes]
# sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# # Print the real and predicted labels
# print(sample_labels)
# print(predicted)

# Display the predictions and the ground truth visually.
# fig = plt.figure(figsize=(10, 10))
# for i in range(len(sample_images)):
#     truth = sample_labels[i]
#     prediction = predicted[i]
#     plt.subplot(5, 2,1+i)
#     plt.axis('off')
#     color='green' if truth == prediction else 'red'
#     plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
#              fontsize=12, color=color)
#     plt.imshow(sample_images[i],  cmap="gray")

# plt.show()

# # Load the test data
# test_images, test_labels = load_data(test_data_directory)

# # Transform the images to 32 by 32 pixels
# test_images32 = [transform.resize(image, (32, 32)) for image in test_images]

# # Convert to grayscale
# # test_images32 = rgb2gray(np.array(test_images32))
# test_images32 = np.array(test_images32)
# # Run predictions against the full test set.
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     predicted = sess.run([correct_pred], feed_dict={x: test_images32})[0]

# # Calculate correct matches 
# match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# # Calculate the accuracy
# accuracy = match_count / len(test_labels)

# # Print the accuracy
# print("Accuracy: {:.3f}".format(accuracy))

# """
# We can able to create a Keras function last_output_fn that obtains a layer’s output (it’s activations),
# given some input data.
# """

# # from keras import backend as K

# # def extract_layer_output(model, layer_name, input_data):
# #   layer_output_fn = K.function([model.layers[0].input],
# #                                [model.get_layer(layer_name).output])

# #   layer_output = layer_output_fn([input_data])

# #   # layer_output.shape is (num_units, num_timesteps)
# #   return layer_output[0]