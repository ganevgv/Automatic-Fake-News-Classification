import sys
import numpy as np
import tensorflow as tf
from import_data import import_tensorflow_data


X_train, y_train, X_valid, y_valid, X_test, y_test = import_tensorflow_data(pos=False, meta_all=False)


model_folder = '../saved_models/rnn_multi_model/'
model_name = model_folder + 'rnn_multi_model.ckpt'
model_text = model_folder + 'rnn_multi_summary.txt'
input_argument = sys.argv[1:]
train_check = ['-train']


max_statement_len = np.shape(X_train['embedded_statement'])[1]
n_classes = np.shape(y_train)[1]

[vocab_size, embedding_size] = np.shape(X_train['W_embeddings'])
W_embed_trainable = True

rnn_size = 64
n_rnn_layers = 2

batch_size = 64
learning_rate = 0.0001
dropout = 0.8
n_epochs = 35
n_batches = np.ceil(np.shape(X_train['embedded_statement'])[0] / batch_size)
tf.set_random_seed(13)


# construct neuron network graph
# placeholders
statement = tf.placeholder(tf.int32, [None, max_statement_len], name='statement')          						# [batch_size x max_statement_len]
statement_len = tf.cast(tf.count_nonzero(statement, axis=1), tf.int32)
label = tf.placeholder(tf.int32, [None, n_classes], name='label')											    # [batch_size x n_classes]
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# word embeddings layer
W_embed = tf.get_variable('W_embed', shape=[vocab_size, embedding_size], initializer=tf.random_uniform_initializer(-0.1, 0.1), trainable=W_embed_trainable)
W_embed = W_embed.assign(np.array(X_train['W_embeddings']))
embedded_statement = tf.nn.embedding_lookup(W_embed, statement)                                                 # [batch_size x max_statement_len x embedding_size]


# stacked bi-LSTM layer
def lstm_cell(rnn_size=rnn_size, keep_prob=keep_prob):
    cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
    return cell


fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size=rnn_size) for _ in range(n_rnn_layers)], state_is_tuple=True)
bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size=rnn_size) for _ in range(n_rnn_layers)], state_is_tuple=True)
_, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_statement, sequence_length=statement_len, dtype=tf.float32)

final_state_fw = tf.concat([final_state[0][i].h for i in range(n_rnn_layers)], axis=-1)
final_state_bw = tf.concat([final_state[1][i].h for i in range(n_rnn_layers)], axis=-1)
final_state = tf.concat([final_state_fw, final_state_bw], axis=1)

# output layer
W_out = tf.get_variable('W_out', shape=[2 * n_rnn_layers * rnn_size, n_classes], initializer=tf.contrib.layers.xavier_initializer())
b_out = tf.get_variable('b_out', shape=n_classes, initializer=tf.contrib.layers.xavier_initializer())

label_pred = tf.add(tf.matmul(final_state, W_out), b_out)

# define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=label_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label_pred, 1), tf.argmax(label, 1)), tf.float32))

# define saver and checkpoint
saver = tf.train.Saver()


# begin session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        epoch_average_loss = 0

        for i in xrange(0, np.shape(X_train['embedded_statement'])[0], batch_size):
            batch_statement = X_train['embedded_statement'][i: i + batch_size, :]
            batch_label = y_train[i: i + batch_size, :]
            feed_dict = {statement: batch_statement, label: batch_label, keep_prob: dropout}

            _, loss = sess.run([optimizer, cross_entropy], feed_dict=feed_dict)
            epoch_average_loss += loss

        # calculate and print Epoch number, Mean Train loss, Train error rate and Test error rate
        print('------- Epoch %d -------' % (epoch))
        print(' Average train loss: %5f' % (epoch_average_loss / n_batches))
        print(' Train accuracy: %5f' % (sess.run(accuracy, {statement: X_train['embedded_statement'], label: y_train, keep_prob: 1.0})))
        print(' Valid accuracy: %5f' % (sess.run(accuracy, {statement: X_valid['embedded_statement'], label: y_valid, keep_prob: 1.0})))
        print(' Test accuracy: %5f' % (sess.run(accuracy, {statement: X_test['embedded_statement'], label: y_test, keep_prob: 1.0})))
