#!/usr/bin/python3
from __future__ import print_function

import sys
import os
import random
import numpy
import tensorflow as tf
from tensorflow.contrib import rnn

def myNormalize(_2ddata):
    return ( _2ddata / numpy.linalg.norm(_2ddata) )

# Argv config
if len(sys.argv) < 5:
    print("\x1b[0;31;40m<Usage> python3 rnn_trigram.py embedding_file training_folder testing_file output_file\x1b[0m")
    sys.exit()
is_load = True if "--load" in sys.argv else False

# Seed
random.seed(3318)
numpy.random.seed(18)
tf.set_random_seed(33)

# Read embedding from argv[1]
word2Idx = {} # diction: key = string, value = ID
embeddings = [] # shape = (rangeOfID, embeddingDimension)
with open(sys.argv[1], "r") as fin:
    for wordIdx, line in enumerate(fin):
        lineData = line.rstrip().split(" ")
        word2Idx[ lineData[0] ] = wordIdx
        embeddings.append( list( map(float, lineData[1:]) ) )
embeddings = numpy.array(embeddings)

# Read training data from argv[2]
data = [] # 2D lists, for example data[1][8] = the ID of the 9th term in the 2nd sentence
count = 0
unknown = 0
for root, dirs, files in os.walk(sys.argv[2], topdown=False):
    for name in files:
        with open(os.path.join(root, name), "r") as fin:
            for line in fin:
                lineData = line.lower().replace(",", "").replace(".", "").replace("\'", " ").rstrip().split()
                terms = []
                for term in lineData:
                    count += 1
                    if term in word2Idx:
                        terms.append( word2Idx[term] )
                    else:
                        unknown += 1
                data.append(terms)

# Print shape and head
print("\x1b[0;31;40m<IO Completed> Data Shape Testing\x1b[0m")
print( "(total, unknown) |", (count, unknown) )
print( "len(word2Idx) |", len(word2Idx) )
print("embeddings.shape |", embeddings.shape)
print("len(data) |", len(data) )
print("word2Idx[\"cup\"] |", word2Idx["cup"])
print("data[0] |", data[0])

# Parameters
learning_rate = 0.01
epoch = 6
batch_size = 4000 # number of sentences in a batch

# Network Parameters
n_dimensions = embeddings.shape[1] # embedding dimensions
n_input = 1 * n_dimensions # data input (sum of two 50-dimensioned embeddings)
n_steps = 10 # number of steps in RNN
n_hidden = 256 # hidden layer num of features
n_output = n_dimensions # output layer (50-dimensioned embedding)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_steps, n_output])

# Define weights and variables
weights = tf.Variable(tf.random_normal([n_hidden, n_output]))
biases = tf.Variable(tf.random_normal([n_output]))

x_reshape = tf.transpose(x, [1, 0, 2])
x_reshape = tf.reshape(x_reshape, [-1, n_input])
x_reshape = tf.split(x_reshape, num_or_size_splits=n_steps, axis=0)

lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.5)

outputs, states = rnn.static_rnn(lstm_cell, x_reshape, dtype=tf.float32)
outputs = tf.stack(outputs)
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = tf.reshape(outputs, [-1, n_hidden])

pred = tf.add(tf.matmul(outputs, weights), biases)
pred_norm = tf.norm(pred, axis=1,keep_dims=True)
pred_unit = pred / pred_norm

# Define loss and optimizer, after reshaping y to (batch_size*n_steps, n_output)
y_reshape = tf.reshape(y,[-1,n_output])
y_norm = tf.norm(y_reshape,axis=1,keep_dims=True)
y_unit = y_reshape / y_norm
cost = -tf.reduce_sum(tf.matmul(tf.reshape(y_unit, [1, -1]),tf.reshape(pred_unit,[-1,1]))) / (batch_size * n_steps)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
print("\x1b[0;31;40m<Launch> RNN Training\x1b[0m")
progress1 = ["[| \x1b[0;32;40m>\x1b[0m", "[| \x1b[0;32;40m>>\x1b[0m", "[| \x1b[0;32;40m>>>\x1b[0m", "[| \x1b[0;32;40m>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>>\x1b[0m"]
progress2 = ["\x1b[0;31;40m---------\x1b[0m |]", "\x1b[0;31;40m--------\x1b[0m |]", "\x1b[0;31;40m-------\x1b[0m |]", "\x1b[0;31;40m------\x1b[0m |]", "\x1b[0;31;40m-----\x1b[0m |]", "\x1b[0;31;40m----\x1b[0m |]", "\x1b[0;31;40m---\x1b[0m |]", "\x1b[0;31;40m--\x1b[0m |]", "\x1b[0;31;40m-\x1b[0m |]", " |]"]
lenData = len(data) # record the number of sentences since the progressing bar needs it
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
saver = tf.train.Saver(tf.all_variables())
with tf.Session(config=config) as sessInv:
    if not is_load:
        sessInv.run(init)
    step = 9999 if is_load else 1
    # Keep training until reach max iterations
    while step <= epoch:
        random.shuffle(data)
        x_batch_inv = numpy.zeros( (batch_size, n_steps, n_input) )
        y_batch_inv = numpy.zeros( (batch_size, n_steps, n_output) )
        curBatch = 0
        for lineNumber, line in enumerate(data):
            steps_batch = len(line) - 1 # maximum steps of the current sentence
            if steps_batch <= n_steps:
                continue
            delta = numpy.random.randint(steps_batch - n_steps + 1) # randomly decide where the head is
            for i in range(n_steps):
                x_batch_inv[curBatch, i, :n_dimensions] = embeddings[ line[-1 - i - delta] ][:]
                y_batch_inv[curBatch, i, :] = embeddings[ line[-1 - i - delta - 1] ][:]
            # Run optimization op (backprop) if the batch is ready
            curBatch += 1
            if curBatch == batch_size:
                sessInv.run(optimizer, feed_dict={x: x_batch_inv, y: y_batch_inv})
                lossInv = sessInv.run(cost, feed_dict={x: x_batch_inv, y: y_batch_inv})
                pro = lineNumber * 10 // lenData # calculate the progressing percentage
                print("\r0%" + progress1[pro] + progress2[pro] + "100% - Epoch: " + str(step) + "/" + str(epoch) + " - Line Number: " + str(lineNumber) + " - Loss(backward): " + str(lossInv), end = "")
                curBatch = 0
        step += 1
        print("")
    if is_load:
        print("\x1b[0;32;40m<Skipping> Loading Model\x1b[0m")
        latest_checkpoint = tf.train.latest_checkpoint("./backward_model/")
        saver = tf.train.import_meta_graph("backward_model/model.meta")
        saver.restore(sessInv, latest_checkpoint)
    else:
        print("\x1b[0;32;40m<Optimization Finished> Saving Model\x1b[0m")
        saver.save(sessInv, "backward_model/model")

# Testing data
    print("\x1b[0;31;40m<Testing> Dealing with Testing Data\x1b[0m")
    dataTest = [] # 2D lists, for example dataTest[1][8] = the ID of the 9th term in the 2nd testing sentence
    ansTest = [] # 2D lists, for example ansTest[6][3] = the ID of the 4th answer in the 7th testing sentence
    idxTest = [] # the indexes of the answers in the sentences
    idxInvTest = [] # the inverse indexes of the answers in the sentences
    with open(sys.argv[3], "r") as fin:
        for lineNumber, line in enumerate(fin):
            if lineNumber == 0:
                continue
            lineData = line.lower().replace("\"", "").replace(".", "").replace("\'", " ").rstrip().split(",")
            contents = " ".join(lineData[1:-5]).replace(",", "").split()
            ansTerms = [ (word2Idx[term] if term in word2Idx else word2Idx["something"]) for term in lineData[-5:] ]
            terms = []
            for term in contents:
                if term == "_____":
                    idxTest.append( len(terms) )
                    terms.append( word2Idx["something"] )
                elif term in word2Idx:
                    terms.append( word2Idx[term] )
            if len(terms) < n_steps + 1:
                for i in range( n_steps + 1 - len(terms) ):
                    terms.append(word2Idx["something"]) # append something to the end of the sentence if needed
            dataTest.append(terms)
            ansTest.append(ansTerms)
            idxInvTest.append(len(terms) - idxTest[-1] - 1)

# Output
    with open(sys.argv[4], "w") as fout:
        for lineNumber, line in enumerate(dataTest):
            lineInv = line[::-1] # inverse the sentence and keep it
            x_batch_inv = numpy.zeros( (1, n_steps, n_input) )
            (startInv, targetInv) = (0, idxInvTest[lineNumber] - 1) if idxInvTest[lineNumber] <= n_steps else (idxInvTest[lineNumber] - n_steps, n_steps -1) # determin where the inverse head/target can be set
            for i in range(n_steps):
                x_batch_inv[0, i, :n_dimensions] = embeddings[ lineInv[i + startInv] ][:]
            y_pred_inv = sessInv.run(pred, feed_dict={x: x_batch_inv})[targetInv, :]
            simInvs = numpy.array([ numpy.dot( myNormalize(embeddings[ans]), myNormalize(y_pred_inv) ) for ans in ansTest[lineNumber] ])
            if lineNumber == 1000:
                print("Backward Predicted Embedding of Line No.", lineNumber + 1, ": [", y_pred_inv[0], ", ... ,", y_pred_inv[-1], "]")
                print("Backward Cosine Similarity between Answers:", simInvs)
            for sim in (targetInv + 1) * simInvs:
                fout.write(str(sim) + " ")
            fout.write("\n")

print("\x1b[0;32;40m<Done> Backward Output File Available\x1b[0m")

