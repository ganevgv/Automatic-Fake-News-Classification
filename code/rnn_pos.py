import sys
import numpy as np
import tensorflow as tf
from classes.import_data import import_tensorflow_data


X_train, y_train, X_valid, y_valid, X_test, y_test = import_tensorflow_data(pos=True, meta_all=False)
#   14    Valid accuracy: 0.274922   Test accuracy: 0.270460

model_folder = '../saved_models/rnn_pos_model/'
model_name = model_folder + 'rnn_pos_model.ckpt'
model_text = model_folder + 'rnn_pos_summary.txt'
input_argument = sys.argv[1:]
train_check = ['-train']


max_statement_len = np.shape(X_train['embedded_statement'])[1]
n_classes = np.shape(y_train)[1]

[vocab_size, embedding_size] = np.shape(X_train['W_embeddings'])
W_embed_trainable = True
[n_pos, pos_size] = np.shape(X_train['W_pos'])
W_pos_trainable = False
combined_size = embedding_size + pos_size

rnn_size = 64

batch_size = 64
learning_rate = 0.0001
dropout = 0.8
n_epochs = 30
n_batches = np.ceil(np.shape(X_train['embedded_statement'])[0] / batch_size)
tf.set_random_seed(14)


# construct neuron network graph
# placeholders
statement_embed = tf.placeholder(tf.int32, [None, max_statement_len], name='statement_emb')         # [batch_size x max_statement_len]
statement_len = tf.cast(tf.count_nonzero(statement_embed, axis=1), tf.int32)
statement_pos = tf.placeholder(tf.int32, [None, max_statement_len], name='statement_pos')           # [batch_size x max_statement_len]
label = tf.placeholder(tf.int32, [None, n_classes], name='label')                                   # [batch_size x n_classes]
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# word embeddings layer
W_embed = tf.get_variable('W_embed', shape=[vocab_size, embedding_size], initializer=tf.random_uniform_initializer(-0.1, 0.1), trainable=W_embed_trainable)
W_pos = tf.get_variable('W_pos', shape=[n_pos, pos_size], initializer=tf.random_uniform_initializer(-0.1, 0.1), trainable=W_pos_trainable)
W_embed = W_embed.assign(np.array(X_train['W_embeddings']))
W_pos = W_pos.assign(np.array(X_train['W_pos']))
embedded_statement = tf.nn.embedding_lookup(W_embed, statement_embed)                               # [batch_size x max_statement_len x embedding_size]
pos_statement = tf.nn.embedding_lookup(W_pos, statement_pos)                                        # [batch_size x max_statement_len x pos_size]

combined_statement = tf.concat([embedded_statement, pos_statement], axis=2)                         # [batch_size x max_statement_len x combined_size]

# bi-LSTM layer
fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
fw_cell = tf.contrib.rnn.DropoutWrapper(cell=fw_cell, output_keep_prob=keep_prob)
bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
bw_cell = tf.contrib.rnn.DropoutWrapper(cell=bw_cell, output_keep_prob=keep_prob)

_, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, combined_statement, sequence_length=statement_len, dtype=tf.float32)
final_state = tf.concat([final_state[0].h, final_state[1].h], axis=1)

# output layer
W_out = tf.get_variable('W_out', shape=[2 * rnn_size, n_classes], initializer=tf.contrib.layers.xavier_initializer())
b_out = tf.get_variable('b_out', shape=n_classes, initializer=tf.contrib.layers.xavier_initializer())

label_pred = tf.add(tf.matmul(final_state, W_out), b_out)                                           # [batch_size x n_classes]

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
                batch_statement_embed = X_train['embedded_statement'][i: i + batch_size, :]
                batch_statement_pos = X_train['pos_statement'][i: i + batch_size, :]
                batch_label = y_train[i: i + batch_size, :]
                feed_dict = {statement_embed: batch_statement_embed, statement_pos: batch_statement_pos, label: batch_label, keep_prob: dropout}

                _, loss = sess.run([optimizer, cross_entropy], feed_dict=feed_dict)
                epoch_average_loss += loss

            # calculate and print Epoch number, Mean Train loss, Train accuracy and Valid accuracy
            epoch_average_loss = epoch_average_loss / n_batches
            train_accuracy = sess.run(accuracy, {statement_embed: X_train['embedded_statement'], statement_pos: X_train['pos_statement'], label: y_train, keep_prob: 1.0})
            valid_accuracy = sess.run(accuracy, {statement_embed: X_valid['embedded_statement'], statement_pos: X_valid['pos_statement'], label: y_valid, keep_prob: 1.0})
            test_accuracy = sess.run(accuracy, {statement_embed: X_test['embedded_statement'], statement_pos: X_test['pos_statement'], label: y_test, keep_prob: 1.0})

            print('--- Epoch {} --- Average train loss: {:.5f}, Train accuracy: {:.5f}, Valid accuracy: {:.5f}, Test accuracy: {:.5f}'
                  .format(epoch, epoch_average_loss, train_accuracy, valid_accuracy, test_accuracy))

            # write to a text file
            with open(model_text, 'a') as text_file:
                text_file.write('--- Epoch {} --- Average train loss: {:.5f}, Train accuracy: {:.5f}, Valid accuracy: {:.5f}, Test accuracy: {:.5f}\n'
                                .format(epoch, epoch_average_loss, train_accuracy, valid_accuracy, test_accuracy))

            # save model if there is improvement
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                print('Save model to: ' + model_name)
                saver.save(sess, model_name)
                with open(model_text, 'a') as text_file:
                    text_file.write('-------Model saved: new best valid accuracy\n')

    # if not load the model
    else:
        print(' Loading the model')
        saver.restore(sess, model_name)

        print(' Train cross entropy: {:.5f}'.format(sess.run(cross_entropy, {statement_embed: X_train['embedded_statement'], statement_pos: X_train['pos_statement'], label: y_train, keep_prob: 1.0})))
        print(' Valid cross entropy: {:.5f}'.format(sess.run(cross_entropy, {statement_embed: X_valid['embedded_statement'], statement_pos: X_valid['pos_statement'], label: y_valid, keep_prob: 1.0})))
        print(' Test cross entropy: {:.5f}'.format(sess.run(cross_entropy, {statement_embed: X_test['embedded_statement'], statement_pos: X_test['pos_statement'], label: y_test, keep_prob: 1.0})))
        print(' Train accuracy: {:.5f}'.format(sess.run(accuracy, {statement_embed: X_train['embedded_statement'], statement_pos: X_train['pos_statement'], label: y_train, keep_prob: 1.0})))
        print(' Valid accuracy: {:.5f}'.format(sess.run(accuracy, {statement_embed: X_valid['embedded_statement'], statement_pos: X_valid['pos_statement'], label: y_valid, keep_prob: 1.0})))
        print(' Test accuracy: {:.5f}'.format(sess.run(accuracy, {statement_embed: X_test['embedded_statement'], statement_pos: X_test['pos_statement'], label: y_test, keep_prob: 1.0})))