import random
import numpy as np
import tensorflow as tf

class S2SModel:

    def __init__(self, _nHidden = 256, _maxLen = 20, _nWords = 4000, _seed = 3318):
        # seed
        random.seed(_seed)
        np.random.seed(_seed)
        tf.set_random_seed(_seed)
        # parameters
        self.nHidden = _nHidden
        self.maxLen = _maxLen
        self.nWords = _nWords
        # variable
        self.embedding = tf.Variable( tf.random_uniform([_nWords, _nHidden], -0.1, 0.1) )
        self.htWeight = tf.Variable( tf.random_uniform([2 * _nHidden, _nHidden], -0.1, 0.1) )
        self.htBias = tf.Variable( tf.zeros([_nHidden]) )
        self.attWeight = tf.Variable( tf.random_uniform([_nHidden, 1], -0.1, 0.1) )
        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(_nHidden)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(_nHidden)
        self.weight = tf.Variable( tf.random_uniform([_nHidden, _nWords], -0.1, 0.1) )
        self.bias = tf.Variable( tf.zeros([_nWords]) )
        # place holder
        self.question = tf.placeholder(tf.int32, [None, _maxLen])
        self.answer = tf.placeholder(tf.int32, [None, _maxLen])
        self.reward = tf.placeholder(tf.float32, [None])

    def buildModel(self, _batchSize = 32):
        # mask
        mask = tf.to_float( tf.not_equal(self.answer, 0) )
        # embedding
        questionFlat = tf.reshape(self.question, [-1])
        questionEmb = tf.reshape(
            tf.nn.embedding_lookup(self.embedding, questionFlat), [-1, self.maxLen, self.nHidden])
        answerFlat = tf.reshape(self.answer, [-1])
        answerEmb = tf.reshape(
            tf.nn.embedding_lookup(self.embedding, answerFlat), [-1, self.maxLen, self.nHidden])
        # memory cells
        c1 = tf.zeros([_batchSize, self.nHidden])
        h1 = tf.zeros([_batchSize, self.nHidden])
        c2 = tf.zeros([_batchSize, self.nHidden])
        h2 = tf.zeros([_batchSize, self.nHidden])
        # zeros
        padding = tf.zeros([_batchSize, self.nHidden])
        # output
        outputs = []
        alphaList = []
        generatedWords = []
        loss = 0.
        # encoding
        for i in range(self.maxLen):
            with tf.variable_scope( "LSTM1", reuse = (i != 0) ):
                output1, (c1, h1) = self.lstm1(questionEmb[:, i, :], state = [c1, h1])
            outputs.append(output1)
            with tf.variable_scope( "LSTM2", reuse = (i != 0) ):
                output2, (c2, h2) = self.lstm2(tf.concat([padding, output1], axis = 1), state = [c2, h2])
        outputs = tf.transpose( tf.stack(outputs), (1, 0, 2) )
        # decoding
        for i in range(self.maxLen - 1):
            curEmb = answerEmb[:, i, :]
            with tf.variable_scope("LSTM1", reuse = True):
                output1, (c1, h1) = self.lstm1(padding, state = [c1, h1])
            # attention
            ht = tf.nn.relu( tf.nn.xw_plus_b(tf.concat([curEmb, output1], axis = 1), self.htWeight, self.htBias) )
            attH = tf.nn.relu( outputs + tf.expand_dims(h2, 1) )
            attOut = tf.reshape(tf.matmul(tf.reshape(attH, [-1, self.nHidden]), self.attWeight), [-1, self.maxLen])
            alpha = tf.nn.softmax(attOut)
            context = tf.reduce_sum(outputs * tf.expand_dims(alpha, 2), axis = 1)
            alphaList.append(alpha)
            with tf.variable_scope("LSTM2", reuse = True):
                output2, (c2, h2) = self.lstm2(tf.concat([ht, context], axis = 1), state = [c2, h2])
            labels = tf.expand_dims(self.answer[:, i + 1], axis = 1)
            indices = tf.expand_dims(tf.range(0, _batchSize, 1), axis = 1)
            concated = tf.concat([indices, labels], axis = 1)
            onehot = tf.sparse_to_dense(concated, np.asarray([_batchSize, self.nWords]), 1.0, 0.0)
            logits = tf.nn.xw_plus_b(output2, self.weight, self.bias)
            crossEn = tf.nn.softmax_cross_entropy_with_logits(labels = onehot, logits = logits) * mask[:, i + 1]
            maxProbIdx = tf.cast(tf.argmax(logits, axis = 1), tf.int32)
            generatedWords.append(maxProbIdx)
            loss = loss + tf.reduce_sum(crossEn * self.reward) / _batchSize
            # alpha regularization
            alphas = tf.transpose(tf.stack(alphaList), perm = [1, 0, 2])
            alphasAll = tf.reduce_sum(alphas, axis = 1)
            alphaReg = 1e-2 * tf.reduce_sum(alphasAll ** 2)
            loss = loss + alphaReg / _batchSize
        # return info
        return loss, generatedWords

    def buildGenerator(self, _batchSize = 32):
        # embedding
        questionFlat = tf.reshape(self.question, [-1])
        questionEmb = tf.reshape(
            tf.nn.embedding_lookup(self.embedding, questionFlat), [-1, self.maxLen, self.nHidden])
        # memory cells
        c1 = tf.zeros([_batchSize, self.nHidden])
        h1 = tf.zeros([_batchSize, self.nHidden])
        c2 = tf.zeros([_batchSize, self.nHidden])
        h2 = tf.zeros([_batchSize, self.nHidden])
        # zeros
        padding = tf.zeros([_batchSize, self.nHidden])
        # output
        outputs = []
        generatedWords = []
        # encoding
        for i in range(self.maxLen):
            with tf.variable_scope("LSTM1", reuse = True):
                output1, (c1, h1) = self.lstm1(questionEmb[:, i, :], state = [c1, h1])
            outputs.append(output1)
            with tf.variable_scope("LSTM2", reuse = True):
                output2, (c2, h2) = self.lstm2(tf.concat([padding, output1], axis = 1), state = [c2, h2])
        outputs = tf.transpose( tf.stack(outputs), (1, 0, 2) )
        # decoding
        maxProbIdx = tf.ones([_batchSize], tf.int32)
        for i in range(self.maxLen - 1):
            curEmb = tf.reshape(
                tf.nn.embedding_lookup( self.embedding, tf.expand_dims(maxProbIdx, 1) ), [-1, self.nHidden])
            with tf.variable_scope("LSTM1", reuse = True):
                output1, (c1, h1) = self.lstm1(padding, state = [c1, h1])
            # attention
            ht = tf.nn.relu( tf.nn.xw_plus_b(tf.concat([curEmb, output1], axis = 1), self.htWeight, self.htBias) )
            attH = tf.nn.relu( outputs + tf.expand_dims(h2, 1) )
            attOut = tf.reshape(tf.matmul(tf.reshape(attH, [-1, self.nHidden]), self.attWeight), [-1, self.maxLen])
            alpha = tf.nn.softmax(attOut)
            context = tf.reduce_sum(outputs * tf.expand_dims(alpha, 2), axis = 1)
            with tf.variable_scope("LSTM2", reuse = True):
                output2, (c2, h2) = self.lstm2(tf.concat([ht, context], axis = 1), state = [c2, h2])
            logits = tf.nn.xw_plus_b(output2, self.weight, self.bias)
            maxProbIdx = tf.argmax(logits, axis = 1)
            generatedWords.append(maxProbIdx)
        # return info
        return generatedWords


