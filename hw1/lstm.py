import tensorflow as tf
from tensorflow.contrib import rnn
# import matplotlib.pyplot as plt
import math, random
import collections
import numpy as np
import sys

random.seed(3318)
np.random.seed(18)
tf.set_random_seed(33)

n_dimensions = 12000 + 1 # embedding dimensions
n_steps = 10
n_output = n_dimensions # output layer (50-dimensioned embedding)
n_sampled = 1000
n_hidden = 256 # hidden layer num of features

# Parameters
learning_rate = 0.01
epoch = 10
batch_size = 1000 # number of sentences in a batch
drop_out_prob = 0.5

is_load = True if "--load" in sys.argv else False

# 0: padding element
# n_dimension - 1: unknown
# '.' and ',' are counted as words
def read_file_build_dict(filename):
    words = []
    with open(filename, 'r') as f:
        for line in f:
            sentence = line.split()
            for word in sentence:
                if word[0] == '\'' and word[len(word) - 1] and len(word) > 2 == '\'':
                    words.append(word[1:-1].lower())
                elif word[0] == '\'' and len(word) > 1:
                    words.append(word[1:].lower())
                elif word[len(word) - 1] == '\'' and len(word) > 1:
                    words.append(word[:-1].lower())
                else:
                    words.append(word.lower())
    print("Complete %s" % filename)
    return words
def build_dict():
    words = []
    for i in range(522):
        words.extend(read_file_build_dict("training_data/%d" % (i+1)))
    count = [['UNK', n_dimensions - 1]]
    count.extend(collections.Counter(words).most_common(n_dimensions - 2))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary) + 1
    return dictionary
dictionary = build_dict()

def read_file(filename):
    data = []
    total_len = 0
    with open(filename, 'r') as f:
        for line in f:
            out = []
            sentence = line.split()
            for word in sentence:
                if word.lower() in dictionary:
                    out.append(dictionary[word.lower()])
            total_len += len(out)
            while(len(out) < n_steps + 1):
                out.append(0)
            data.append(out)
    return data, total_len

def build_dataset():
    dataset = []
    total_len = 0
    for i in range(522):
        data, length = read_file("training_data/%d" % (i+1))
        dataset.extend(data)
        total_len += length
    print("Complete dataset")
    return dataset, total_len

dataset, total_len = build_dataset()

x = tf.placeholder(tf.int32, [None, n_steps])
y = tf.placeholder(tf.int32, [None, n_steps, 1])
drop_out = tf.placeholder(tf.float32)
# lengths = tf.placeholder(tf.int32, [None])
onehot_x = tf.one_hot(x, depth=n_dimensions, axis=-1) #return batch x features x depth

lstm_cell = rnn.LSTMCell(n_hidden)
# lstm_cell = rnn.LSTMCell(n_hidden, use_peepholes=True)
cell = rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=drop_out)
cell = rnn.MultiRNNCell(cells=[cell] * 2, state_is_tuple=True)
outputs, last_states = tf.nn.dynamic_rnn(
    cell=lstm_cell,
    dtype=tf.float32,
  #  sequence_length=lengths,
    inputs=onehot_x)

weights = tf.Variable(tf.random_normal([n_output, n_hidden]))
biases = tf.Variable(tf.zeros([n_output]))

outputs = tf.reshape(outputs, [-1, n_hidden])
labels = tf.reshape(y, [-1, 1])

# loss = tf.reduce_mean(
#       tf.nn.sampled_softmax_loss(weights=weights,
#                      biases=biases,
#                      labels=labels,
#                      inputs=outputs,
#                      num_sampled=n_sampled,
#                      num_classes=n_output,
#                      num_true=1))
softmax = tf.matmul(outputs, tf.transpose(weights)) + biases
pred = tf.nn.softmax(softmax)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labels, [-1]), logits=softmax))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
print("\x1b[0;31;40m<Launch> RNN Training\x1b[0m")
progress1 = ["[| \x1b[0;32;40m>\x1b[0m", "[| \x1b[0;32;40m>>\x1b[0m", "[| \x1b[0;32;40m>>>\x1b[0m", "[| \x1b[0;32;40m>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>>\x1b[0m"]
progress2 = ["\x1b[0;31;40m---------\x1b[0m |]", "\x1b[0;31;40m--------\x1b[0m |]", "\x1b[0;31;40m-------\x1b[0m |]", "\x1b[0;31;40m------\x1b[0m |]", "\x1b[0;31;40m-----\x1b[0m |]", "\x1b[0;31;40m----\x1b[0m |]", "\x1b[0;31;40m---\x1b[0m |]", "\x1b[0;31;40m--\x1b[0m |]", "\x1b[0;31;40m-\x1b[0m |]", " |]"]
lenData = len(dataset)

saver = tf.train.Saver(tf.all_variables())
with tf.Session() as sess:
    if not is_load:
        sess.run(init)
    step = 1
    while step <= epoch:
        if is_load:
            break
        random.shuffle(dataset)
        x_batch = np.zeros( (batch_size, n_steps) )
        y_batch = np.zeros( (batch_size, n_steps, 1) )
        # len_batch = np.zeros((batch_size))
        curBatch = 0
        for lineNumber, line in enumerate(dataset):
            steps_batch = len(line) - 1 # maximum steps of the current sentence
            # len_batch[ curBatch ] = steps_batch
            delta = np.random.randint(steps_batch - n_steps + 1) if (steps_batch - n_steps >= 0) else 0 # randomly decide where the head is
            for i in range(n_steps):
                # x_batch[curBatch, i, :n_dimensions] = np.zeros(n_dimensions)
                # x_batch[ curBatch, i, line[i + delta] ] = 1
                x_batch[curBatch, i] = line[i + delta]
                y_batch[curBatch, i, 0] = line[i + delta + 1]
            # Run optimization op (backprop) if the batch is ready
            curBatch += 1
            if curBatch == batch_size:
                sess.run(optimizer, feed_dict={x: x_batch, y: y_batch, drop_out: drop_out_prob})
                current_loss = sess.run(loss, feed_dict={x: x_batch, y: y_batch, drop_out: 1.0})
                pro = lineNumber * 10 // lenData # calculate the progressing percentage
                print("\r0%" + progress1[pro] + progress2[pro] + "100% - Epoch: " + str(step) + "/" + str(epoch) + " - Line Number: " + str(lineNumber) + " - Loss(forward): " + str(current_loss))
                curBatch = 0
        step += 1
        print("")
    if is_load:
        print("\x1b[0;32;40m<Skipping> Loading Model\x1b[0m")
        latest_checkpoint = tf.train.latest_checkpoint("./model/")
        saver = tf.train.import_meta_graph("model/model.meta")
        saver.restore(sess, latest_checkpoint)
        b = sess.run(biases)
        print(b)
    else:
        print("\x1b[0;32;40m<Optimization Finished> Saving Model\x1b[0m")
        saver.save(sess, "model/model")

    print("\x1b[0;31;40m<Testing> Dealing with Testing Data\x1b[0m")

    dataTest = [] # 2D lists, for example dataTest[1][8] = the ID of the 9th term in the 2nd testing sentence
    ansTest = [] # 2D lists, for example ansTest[6][3] = the ID of the 4th answer in the 7th testing sentence
    idxTest = [] # the indexes of the answers in the sentences
    with open(sys.argv[1], "r") as fin:
        for lineNumber, line in enumerate(fin):
            if lineNumber == 0:
                continue
            lineData = line.lower().replace("\"", "").replace(".", " .").rstrip().split(",")
            raw_contents = " ".join(lineData[1:-5]).replace(",", "").split(" ")
            contents = []
            for term in raw_contents:
              if term == "_____" or term in dictionary:
                contents.append(term)
            ansTerms = [ (dictionary[term] if term in dictionary else n_dimensions - 1) for term in lineData[-5:] ]
            terms = [(dictionary[term] if term in dictionary else n_dimensions - 1) for term in contents]
            # idx = n_dimensions - 1 for unkown words
            if len(terms) < n_steps + 1:
                for i in range( n_steps + 1 - len(terms) ):
                    terms.append(0) # padding
            dataTest.append(terms)
            ansTest.append(ansTerms)
            idxTest.append( contents.index("_____") )

    labels = ["a", "b", "c", "d", "e"]
    with open(sys.argv[2], "w") as fout:
        fout.write("id,answer\n")
        for lineNumber, line in enumerate(dataTest):
            x_batch = np.zeros( (1, n_steps) )
            x_batch_cand = np.zeros( (1, n_steps) )
            (start, target) = (0, idxTest[lineNumber] - 1) if idxTest[lineNumber] <= n_steps else (idxTest[lineNumber] - n_steps, n_steps - 1) # determine where the head/target can be set
            (start_cand, target_cand) = (0, idxTest[lineNumber]) if idxTest[lineNumber] < n_steps else (idxTest[lineNumber] - n_steps + 1, n_steps - 1) # determine where the head/target can be set
            candidate_ans = line[target_cand + 1] if target_cand + 1 < len(line) else 0
            for i in range(n_steps):
                # x_batch[0, i, :n_dimensions] = np.zeros(n_dimensions)
                # x_batch[0, i, line[ i + start ]] = 1
                x_batch[0, i] = line[ i + start ]
                if i < target_cand:
                    x_batch_cand[0, i] = line[ i + start_cand ]
            y_pred = sess.run(pred, feed_dict={x: x_batch, drop_out: 1.0})[target, :]
            sims = np.array([ y_pred[ans] for ans in ansTest[lineNumber] ])

            y_pred_cand = []
            for i, candidate in enumerate(ansTest[lineNumber]):
                x_batch_cand[0, target_cand] = candidate
                y_pred_cand.append(sess.run(pred, feed_dict={x: x_batch_cand, drop_out: 1.0})[target_cand, candidate_ans])

            if lineNumber == 1000:
                # print("Forward Predicted Embedding of Line No.", lineNumber + 1, ": [", y_pred[0], ", ... ,", y_pred[-1], "]")
                print("Forward Probability of Answers:", sims)
            label = labels[np.argmax(np.multiply(y_pred_cand, sims))]
            fout.write(str(lineNumber + 1) + "," + label + "\n")
print("Average length", total_len * 1.0/ len(dataset) )
print("\x1b[0;32;40m<Done> Output File Available\x1b[0m")
