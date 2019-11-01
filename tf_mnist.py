# from tensorflow.examples.tutorials.mnist import input_data
# import datalib
# (X_train, Y_train, X_cv, Y_cv, X_test, Y_test) = 
# csv_data = datalib.load({url: '/home/mohan/Downloads/Automobile_data.csv'})
# #datalib.get_data_from_file(file_name='./f_1d_cos_no_noise_data.npz')
# (N_train,D) = X_train.shape
# D1 = 24
# (N_test,D_out) = Y_test.shape
# W1 = tf.Variable( tf.truncated_normal([D,D1], mean=0.0, stddev=std), name='W1') # (D x D1)
# S1 = tf.Variable( tf.constant(100.0, shape=[]), name='S1') # (1 x 1)
# C1 = tf.Variable( tf.truncated_normal([D1,1], mean=0.0, stddev=0.1), name='C1' ) # (D1 x 1)
# W1_hist = tf.summary.histogram("W1", W1)
# S1_scalar_summary = tf.summary.scalar("S1", S1)
# C1_hist = tf.summary.histogram("C1", C1)


# import input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import matplotlib.pyplot as plt
# Set parameters
image_size = 28
num_classes = 10
learning_rate = 0.01
training_iteration = 30
batch_size = 100
dropout = 0.5
display_step = 2

# TF graph input
x_s = tf.placeholder(tf.float32, [None, image_size*image_size*1])# mnist data image of shape 28*28=784
y_s = tf.placeholder(tf.float32, [None, num_classes]) # 0-9 digits recognition => 10 classes
keep_prob = tf.placeholder(tf.float32)

print(mnist.train.labels.shape)
# plt.hist(mnist.train.labels,10)
# plt.show()

x = tf.reshape(x_s, [-1,28,28,1])

# Create a model
# Set model weights
# W = tf.Variable(tf.truncated_normal([image_size*image_size, labels],stddev=0.1))
# b = tf.Variable(tf.constant(0.1,shape=[labels]))
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
# with tf.name_scope('Wx_b') as scope:
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'w1': tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'w2': tf.Variable(tf.random_normal([3, 3, 32, 64],stddev=0.1)),
        # 5x5 conv, 32 inputs, 64 outputs
    'w3': tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wf1': tf.Variable(tf.random_normal([25*25*64, 1024],stddev=0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes],stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.constant(0.0, shape=[32])),
    'b2': tf.Variable(tf.constant(0.0, shape=[64])),
    'b3': tf.Variable(tf.constant(0.0, shape=[64])),
    'bf1': tf.Variable(tf.constant(0.0, shape=[1024])),
    'out1': tf.Variable(tf.constant(0.0, shape=[num_classes]))
}


# # Creating model


def conv_net(x, weights, biases, dropout):

    # Convolution Layer
    conv1 = conv2d(x, weights['w1'], biases['b1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['w2'], biases['b2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # Convolution Layer
    conv3 = conv2d(conv2, weights['w3'], biases['b3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wf1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wf1']), biases['bf1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out1'])
    return out


# with tf.name_scope("Wx_b") as scope:
#     # Construct a linear model
#     output = tf.matmul(x_s,W)+b
#     #model = tf.nn.softmax(tf.matmul(x_s,W)+b) # Softmax
# # Add summary ops to collect data
# w_h = tf.summary.histogram("weights", W)
# b_h = tf.summary.histogram("biases", b)

logits = conv_net(x, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)
# More name scopes will clean up graph representation
with tf.name_scope("loss") as scope:
    # Minimize error using cross entropy
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_s,logits=logits))

    tf.summary.scalar("loss",loss_op)

with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)    # Create a summary to monitor the cost function
    #tf.summary.scalar("optimizer",train_op)

with tf.name_scope("Accuracy") as scope:
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_s, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy",accuracy)

# Initializing the variables
init = tf.global_variables_initializer()
# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# save_file = './saved_models1/tensorflow_model_0.ckpt' 
# saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Set the logs writer to the folder /tmp/tensorflow_logs
    #merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/home/mohan/logs',tf.get_default_graph())
    # Training cycle
    for iteration in range(training_iteration):
        print('started')
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # print('do it')
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # Fit training using batch data
            # print(batch_xs)
            # print(batch_ys)
            sess.run(train_op, feed_dict={x_s: batch_xs, y_s: batch_ys,keep_prob:0.5})
            # Compute the average loss
            # print('gy')
            avg_cost += sess.run(loss, feed_dict={x_s: batch_xs, y_s: batch_ys, keep_prob:0.5})/total_batch
            # Write logs for each iteration
            # print(avg_cost)
            summary_str = sess.run(merged_summary_op, feed_dict={x_s: batch_xs, y_s: batch_ys, keep_prob:0.5})
            # print('fg')
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
            # Print the accuracy progress on the batch every 100 steps
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x_s:batch_xs,y_s:batch_ys})
                print("Step %d, loss %g training batch accuracy %g %%"%(i,avg_cost, train_accuracy*100))
            save_path = saver.save(sess, save_file)

        # Display logs per iteration step
        if iteration % display_step == 0:
            print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
        writer.close()
    print("Step %d, training batch accuracy: %g %%  "%(i, train_accuracy*100))

    print ("Tuning completed!")
    # Test the model
    #predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y_s, 1))
    # Calculate accuracy
    #val_accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    val_accuracy = accuracy.eval(feed_dict={x_s: mnist.test.images, y_s: mnist.test.labels})
    print( "Validation Accuracy: %g %%  "%(val_accuracy*100))
