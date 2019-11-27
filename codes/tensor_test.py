import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage
import mlflow
import mlflow.sklearn

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

# mlflow.log_param("images",images)
# mlflow.log_param("labels",labels)
# Import the `transform` module from `skimage`
from skimage import transform 

# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]

# images28 = np.array(images28)

#While converting to grayscale images we have to convert the list images28 to array again,
# as the function rgb2gray will take array as an argument

# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray
# Convert `images28` to an array
images28 = np.array(images28)
# Convert `images28` to grayscale
images28 = rgb2gray(images28)

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28,28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
with tf.name_scope("Wx_b") as scope:
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    tf.summary.scalar("logits",logits)
# Define a loss function
with tf.name_scope("loss") as scope:
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)


# Initializing the variables
init = tf.global_variables_initializer()
# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# save_file = './saved_models1/tensorflow_model_0.ckpt' 
# saver = tf.train.Saver()
mlflow.log_metric("loss",loss)
mlflow.log_metric("train_op",train_op)
# mlflow

tf.set_random_seed(1234)
import time
with tf.Session() as sess:
    start_time = time.time()
    sess.run(init)
    summary_writer = tf.summary.FileWriter('/home/mohan/logs',tf.get_default_graph())
    for i in range(400):
        print('EPOCH', i)
        _,Accuracy, loss_val = sess.run([train_op,accuracy, loss], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss_val)
            print("Accuracy: ", Accuracy)
        print('DONE WITH EPOCH')
    end_time = time.time()
    diff_time = end_time - start_time
    print("time_taken : ",diff_time)

# #Prediction

# import random

# # Pick 10 random images
# sample_indexes = random.sample(range(len(images28)), 10)
# sample_images = [images28[i] for i in sample_indexes]
# sample_labels = [labels[i] for i in sample_indexes]

# # Run the "correct_pred" operation
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# # Print the real and predicted labels
# print(sample_labels)
# print(predicted)

# # Display the predictions and the ground truth visually.
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

# # Transform the images to 28 by 28 pixels
# test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# # Convert to grayscale
# test_images28 = rgb2gray(np.array(test_images28))

# # Run predictions against the full test set.
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# # Calculate correct matches 
# match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# # Calculate the accuracy
# accuracy = match_count / len(test_labels)

# Print the accuracy
# print("Accuracy: {:.3f}".format(accuracy))