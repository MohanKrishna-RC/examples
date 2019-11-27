from tensorflow.examples.tutorials.mnist import input_data
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

import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
# Set parameters
image_size = 28
labels = 10
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2
# TF graph input
x_s = tf.placeholder(tf.float32, [None, image_size*image_size*1]) # mnist data image of shape 28*28=784
y_s = tf.placeholder(tf.float32, [None, labels]) # 0-9 digits recognition => 10 classes
keep_prob = tf.placeholder(tf.float32)
# Create a model
# Set model weights
W = tf.Variable(tf.truncated_normal([image_size*image_size, labels],stddev=0.1))
b = tf.Variable(tf.constant(0.1,shape=[labels]))

with tf.name_scope("Wx_b") as scope:
    # Construct a linear model
    output = tf.matmul(x_s,W)+b
    #model = tf.nn.softmax(tf.matmul(x_s,W)+b) # Softmax
# Add summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)
# More name scopes will clean up graph representation
with tf.name_scope("loss") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    # cost_function = -tf.reduce_sum(y_s*tf.log(model))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_s,logits=output))
    # Create a summary to monitor the cost function
    # tf.summary.scalar("cost_function", cost_function)
    tf.summary.scalar("loss",loss)
with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.name_scope("Accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_s, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("Accuracy",accuracy)
# Initializing the variables
init = tf.global_variables_initializer()
# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Set the logs writer to the folder /tmp/tensorflow_logs
    #merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/home/mohan/logs', graph = sess.graph)
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
            sess.run(optimizer, feed_dict={x_s: batch_xs, y_s: batch_ys,keep_prob:0.5})
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
        # if i%100 == 0:
        #     train_accuracy = accuracy.eval(feed_dict=feed_dict)
        #     print("Step %d, training batch accuracy %g %%"%(i, train_accuracy*100))
        # Display logs per iteration step
        if iteration % display_step == 0:
            print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
        
    print("Step %d, training batch accuracy: %g %%  "%(i, train_accuracy*100))

    print ("Tuning completed!")
    # Test the model
    #predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y_s, 1))
    # Calculate accuracy
    #val_accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    val_accuracy = accuracy.eval(feed_dict={x_s: mnist.test.images, y_s: mnist.test.labels})
    print( "Validation Accuracy: %g %%  "%(val_accuracy*100))

# import losswise

# losswise.set_api_key("project api key")
# session = losswise.Session(tag='my_special_lstm', max_iter=10)
# loss_graph = session.graph('loss', kind='min')

# # train an iteration of your model...
# loss_graph.append(x, {'train_loss': train_loss, 'validation_loss': validation_loss})
# # keep training model...

# session.done()