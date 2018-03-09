import sys
import numpy as np
import tensorflow as tf
from classes.import_data import import_tensorflow_data


X_train, y_train, X_valid, y_valid, X_test, y_test = import_tensorflow_data(pos=False, meta_all=False)
#   11   [2]    Valid accuracy: 0.284268    Test accuracy: 0.269680

model_folder = '../saved_models/cnn_w2_model/'
model_name = model_folder + 'cnn_w2_model.ckpt'
model_text = model_folder + 'cnn_w2_summary.txt'
input_argument = sys.argv[1:]
train_check = ['-train']


max_statement_len = np.shape(X_train['embedded_statement'])[1]
n_classes = np.shape(y_train)[1]

[vocab_size, embedding_size] = np.shape(X_train['W_embeddings'])

filter_sizes = [2]
n_channels = 1
n_filters = 128

batch_size = 64
learning_rate = 0.0001
dropout = 0.8
n_epochs = 30
n_batches = np.ceil(np.shape(X_train['embedded_statement'])[0] / batch_size)
tf.set_random_seed(11)


# construct neuron network graph
# placeholders
statement = tf.placeholder(tf.int32, [None, max_statement_len], name='statement')                   # [batch_size x max_statement_len]
label = tf.placeholder(tf.int32, [None, n_classes], name='label')                                   # [batch_size x n_classes]
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# word embeddings layer
W_embed_train = tf.get_variable('W_embed_train', [vocab_size, embedding_size], initializer=tf.random_uniform_initializer(-0.1, 0.1), trainable=True)
W_embed_nottrain = tf.get_variable('W_embed_nottrain', [vocab_size, embedding_size], initializer=tf.random_uniform_initializer(-0.1, 0.1), trainable=False)
W_embed_train = W_embed_train.assign(np.array(X_train['W_embeddings']))
W_embed_nottrain = W_embed_nottrain.assign(np.array(X_train['W_embeddings']))
embedded_statement_ch1 = tf.nn.embedding_lookup(W_embed_train, statement)                           # [batch_size x max_statement_len x embedding_size]
embedded_statement_ch2 = tf.nn.embedding_lookup(W_embed_nottrain, statement)                        # [batch_size x max_statement_len x embedding_size]
embedded_statement_ch1 = tf.reshape(embedded_statement_ch1, shape=[-1, max_statement_len, embedding_size, n_channels])
#                                                                                                   # [batch_size x max_statement_len x embedding_size x n_channels]
embedded_statement_ch2 = tf.reshape(embedded_statement_ch2, shape=[-1, max_statement_len, embedding_size, n_channels])
#                                                                                                   # [batch_size x max_statement_len x embedding_size x n_channels]
embedded_statement = tf.concat([embedded_statement_ch1, embedded_statement_ch2], 3)


# convolutional and max-pool layer
pooled_outputs = []
for index, filter_size in enumerate(filter_sizes):
    W_conv = tf.get_variable('W_conv_{}'.format(index), shape=[filter_size, embedding_size, 2 * n_channels, n_filters], initializer=tf.contrib.layers.xavier_initializer())
    b_conv = tf.get_variable('b_conv_{}'.format(index), shape=[n_filters], initializer=tf.contrib.layers.xavier_initializer())

    conv_layer = tf.nn.relu(tf.add(tf.nn.conv2d(embedded_statement, W_conv, strides=[1, 1, 1, 1], padding='VALID'), b_conv))
    #                                                                                               # [batch_size x max_statement_len - filter_size + 1 x 1 x n_filters]
    conv_layer = tf.nn.max_pool(conv_layer, ksize=[1, max_statement_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    #                                                                                               # [batch_size x 1 x 1 x n_filters]
    pooled_outputs.append(conv_layer)

# flatten layer with dropout
flat_layer = tf.concat(pooled_outputs, 3)                                                           # [batch_size x 1 x 1 x n_filters * len(filter_sizes)]
dropout_layer = tf.nn.dropout(tf.squeeze(flat_layer), keep_prob=keep_prob)                          # [batch_size x n_filters * len(filter_sizes)]

# output layer
W_out = tf.get_variable('W_out', shape=[n_filters * len(filter_sizes), n_classes], initializer=tf.contrib.layers.xavier_initializer())
b_out = tf.get_variable('b_out', shape=[n_classes], initializer=tf.contrib.layers.xavier_initializer())
label_pred = tf.add(tf.matmul(dropout_layer, W_out), b_out)                                         # [batch_size x n_classes]

# define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=label_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label_pred, 1), tf.argmax(label, 1)), tf.float32))

# define saver and checkpoint
saver = tf.train.Saver()


# begin session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # check if -train specified
    if input_argument == train_check:
        print(' Training the model ')
        best_valid_accuracy = 0

        for epoch in range(n_epochs):
            epoch_average_loss = 0

            for i in xrange(0, np.shape(X_train['embedded_statement'])[0], batch_size):
                batch_statement = X_train['embedded_statement'][i: i + batch_size, :]
                batch_label = y_train[i: i + batch_size, :]
                feed_dict = {statement: batch_statement, label: batch_label, keep_prob: dropout}

                _, loss = sess.run([optimizer, cross_entropy], feed_dict=feed_dict)
                epoch_average_loss += loss

            # calculate and print Epoch number, Mean Train loss, Train accuracy and Valid accuracy
            epoch_average_loss = epoch_average_loss / n_batches
            train_accuracy = sess.run(accuracy, {statement: X_train['embedded_statement'], label: y_train, keep_prob: 1.0})
            valid_accuracy = sess.run(accuracy, {statement: X_valid['embedded_statement'], label: y_valid, keep_prob: 1.0})
            test_accuracy = sess.run(accuracy, {statement: X_test['embedded_statement'], label: y_test, keep_prob: 1.0})

            print('--- Epoch {} --- Average train loss: {:.5f}, Train accuracy: {:.5f}, Valid accuracy: {:.5f}, Test accuracy: {:.5f}'
                  .format(epoch, epoch_average_loss, train_accuracy, valid_accuracy, test_accuracy))

            # # write to a text file
            # with open(model_text, 'a') as text_file:
            #     text_file.write('--- Epoch {} --- Average train loss: {:.5f}, Train accuracy: {:.5f}, Valid accuracy: {:.5f}, Test accuracy: {:.5f}\n'
            #                     .format(epoch, epoch_average_loss, train_accuracy, valid_accuracy, test_accuracy))

            # # save model if there is improvement
            # if valid_accuracy > best_valid_accuracy:
            #     best_valid_accuracy = valid_accuracy
            #     print('Save model to: ' + model_name)
            #     saver.save(sess, model_name)
            #     with open(model_text, 'a') as text_file:
            #         text_file.write('-------Model saved: new best valid accuracy\n')

    # if not load the model
    else:
        print(' Loading the model')
        saver.restore(sess, model_name)

        print(' Train cross entropy: {:.5f}'.format(sess.run(cross_entropy, {statement: X_train['embedded_statement'], label: y_train, keep_prob: 1.0})))
        print(' Valid cross entropy: {:.5f}'.format(sess.run(cross_entropy, {statement: X_valid['embedded_statement'], label: y_valid, keep_prob: 1.0})))
        print(' Test cross entropy: {:.5f}'.format(sess.run(cross_entropy, {statement: X_test['embedded_statement'], label: y_test, keep_prob: 1.0})))
        print(' Train accuracy: {:.5f}'.format(sess.run(accuracy, {statement: X_train['embedded_statement'], label: y_train, keep_prob: 1.0})))
        print(' Valid accuracy: {:.5f}'.format(sess.run(accuracy, {statement: X_valid['embedded_statement'], label: y_valid, keep_prob: 1.0})))
        print(' Test accuracy: {:.5f}'.format(sess.run(accuracy, {statement: X_test['embedded_statement'], label: y_test, keep_prob: 1.0})))
