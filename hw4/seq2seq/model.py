# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import math, os, sys, random
from preprocessing import special_word_to_id

class seq2seq:
    def __init__(self,
                n_layers    = 4,
                n_hidden    = 256,
                n_step1        = 20,
                n_step2        = 20,
                dim_input    = 5000,
                dim_output    = 5000,
                use_ss        = False,
                use_att        = False,
                use_bn        = False,
                use_dropout = True,
                beam_size    = 1,
                alpha_c        = 0.0,
                seed        = 3318):
        """
        parameters
        ----------
        n_layers        integer, number of layers
        n_hidden        integer, number of hidden dimension of each layer
        n_steps1        integer, n_steps of the encoding LSTM
        n_steps2        integer, n_steps of the decoding LSTM
        dim_input        integer, input dimension
        dim_output        integer, output dimension
        use_ss          boolean, whether to use schedule sampling
        use_att         boolean, whether to use attention
        use_bn          boolean, whether to use batch normalization
        use_dropout        boolean, whether to use dropout in training
        beam_size       integer, number of beams
        alpha_c         integer, regularization parameters for attention
        seed            integer, ramdom seed
        """

        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.n_layers   = n_layers
        self.n_hidden    = n_hidden
        self.n_step1    = n_step1
        self.n_step2    = n_step2
        self.dim_input    = dim_input
        self.dim_output    = dim_output
        self.use_ss     = use_ss
        self.use_att    = use_att
        self.use_bn     = use_bn
        self.use_dropout = use_dropout
        self.beam_size  = beam_size
        self.alpha_c    = alpha_c
        self.seed       = seed

        ### Placeholders and Variables
        self.input_text = tf.placeholder(tf.int32, [None, self.n_step1])
        self.output_text = tf.placeholder(tf.int32, [None, self.n_step2])

        # sampling rate should be from large to small while training
        # the probability of using the true label for decoder's input
        self.schedule_sampling_rate = tf.placeholder(tf.float32)

        if self.use_dropout:
            self.dropout_rate = tf.placeholder(tf.float32)
            self.encoder = self.multi_lstm('lstm1', dropout_rate=self.dropout_rate)
            self.decoder = self.multi_lstm('lstm2', dropout_rate=self.dropout_rate)
        else:
            self.encoder = self.multi_lstm('lstm1', dropout_rate=None)
            self.decoder = self.multi_lstm('lstm2', dropout_rate=None)
        print(self.encoder)
        print(self.decoder)
        if self.use_att:
            # TODO: build attention
            1 + 1
        self.input_embed_W = tf.get_variable('input_embed_W', [dim_input, n_hidden], tf.float32, initializer=tf.random_normal_initializer())
        # self.input_embed_b = tf.get_variable('input_embed_b', [n_hidden], tf.float32, initializer=tf.constant_initializer(0.0))
        self.output_embed_W = tf.get_variable('output_embed_W', [dim_output, n_hidden], tf.float32, initializer=tf.random_normal_initializer())

        # NOTE: input embedding lookup without b or with b
        self.embed_input = tf.nn.embedding_lookup(self.input_embed_W, self.input_text)
        self.embed_output = tf.nn.embedding_lookup(self.output_embed_W, self.output_text)

        self.output_proj_W = tf.get_variable('output_proj_W', [n_hidden, dim_output], tf.float32, initializer=tf.random_normal_initializer())
        self.output_proj_b = tf.get_variable('output_proj_b', [dim_output], tf.float32, initializer=tf.constant_initializer(0.0))

    def _sample(self, tensor1, tensor2, prob1):
        """
        p(tensor1) = prob1
        p(tensor2) = (1 - prob1)
        """
        return tf.cond(tf.random_uniform([1])[0] >= prob1, lambda: tensor2, lambda: tensor1)


    def build_model(self, batch_size=64, is_training=True):
        ### Encoding
        encoder_outputs = []
        self.initial_state = self.encoder.zero_state(batch_size, tf.float32)
        state = self.initial_state
        # print(state)
        with tf.variable_scope("Encoder") as scope:
            ## TODO: the state of multi_lstm is n-tuple
            # state = self.multi_lstm('none').zero_state(batch_size, tf.float32)
            for i in range(0, self.n_step1):
                print('step' + str(i))
                if i > 0:
                    scope.reuse_variables()
                    print(scope)
                else:
                    # initial state
                    print(state)
                output, state = self.encoder(self.embed_input[:, i, :], state=state)

                encoder_outputs.append(output)

        ### Decoding
        loss = 0.0
        decoder_outputs = []
        generated_words = []

        # TODO: 生好mask
        mask = tf.to_float(tf.not_equal(self.output_text, special_word_to_id('<PAD>'))) # <PAD> 0

        with tf.variable_scope("Decoder") as scope:
            for i in range(self.n_step2 - 1):
                if i > 0:
                    scope.reuse_variables()
                    if is_training:
                        if self.use_ss:
                            # TODO schedule sampling
                            sample_tensor = self._sample(self.embed_output[:,i,:], predict_word_embed, self.schedule_sampling_rate)
                            output, state = self.decoder(sample_tensor, state=state)
                        else:
                            output, state = self.decoder(self.embed_output[:,i,:], state=state)
                    else: # testing
                        output, state = self.decoder(predict_word_embed, state=state)
                else: # i == 0
                    if is_training:
                        output, state = self.decoder(self.embed_output[:,i,:], state=state)
                    else: # testing # TODO: testing start HOWTO <BOS> 2
                        output, state = self.decoder(tf.nn.embedding_lookup(self.output_embed_W, [special_word_to_id('<BOS>') for _ in range(batch_size)]), state=state)

                # TODO: NOT SURE the which output is the last layer
                logits = tf.matmul(output, self.output_proj_W) + self.output_proj_b
                # decoder_outputs.append(logits)
                max_prob_index = tf.argmax(logits, axis=1)
                generated_words.append(max_prob_index)

                predict_word_embed = tf.nn.embedding_lookup(self.output_embed_W, max_prob_index)

                if is_training:
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_text[:, i+1], logits=logits)
                    cross_entropy = cross_entropy * mask[:, i]

                    loss = loss + (tf.reduce_sum(cross_entropy) / batch_size)



        return loss, tf.transpose(generated_words)
    def train(self, Data, batch_size=64, learning_rate=1e-3, epoch=100, period=3, name='model', dropout_rate=0.0, resume_model_path=None):
        """
        Parameters
        ----------
        X               shape (# samples, 80, 4096)
        y               shape (# samples, length of sentences)
        valid_X         shape (# validation samples, 80, 4096)
        valid_y         shape (# validation samples, length of sentences)
        Data            instance of constructed data class
        batch_size      integer, number of samples in a batch
        learning_rate   float
        epoch           integer, number of epochs to run
        period          integer, intervals between checkpoints
        name            string, model name to save
        dropout_rate    float
        resume_model_path   the path of resume path, otherwise, None
        """
        # TODO: 如果是restore就不intialize

        X, y, valid_X, valid_y = Data.gen_train_data(test_ratio=0.01)

        os.makedirs(os.path.join('./models/', name), exist_ok=True)
        with tf.variable_scope('Train') as scope:
            loss, _ = self.build_model(batch_size=batch_size, is_training=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            scope.reuse_variables()
            valid_loss, _ = self.build_model(batch_size=len(valid_X), is_training=True)
            _, sample_sentence = self.build_model(batch_size=5, is_training=False)

            # with tf.variable_scope(tf.get_variable_scope(),reuse=False):
            #     # TODO: global_step在哪裡加
            #     global_step = tf.Variable(0, trainable=False)
            #     boundaries = [int(epoch*0.5), int(epoch*0.9)]
            #     values = [learning_rate, learning_rate/3, learning_rate/10]
            #     lr = tf.train.piecewise_constant(global_step, boundaries, values)
            #
            #     optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            #     saver = tf.train.Saver(max_to_keep=10)
            sample_rates = [1, 0.8, 0.5]
            sample_rate_boundary = [epoch / 2, epoch * 3 / 4, epoch]
            # TODO: gradient clipping
            saver = tf.train.Saver(max_to_keep=10)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                if resume_model_path != None:
                    saver.restore(sess, resume_model_path)
                    print('Restore model done')

                for step in range(epoch):

                    for i in range(len(sample_rates)):
                        if step <= sample_rate_boundary[i]:
                            schedule_sampling_rate = sample_rates[i]
                            break
                    print("The schedule sampling rate is {}".format(schedule_sampling_rate))
                    # TODO: valid_X is not the size of (batch_size * n_hidden), so will cause ERROR
                    if step % period == 0:
                        save_path = saver.save(sess, "./models/{}/model_after_epoch_{}.ckpt".format(name,step))
                        # saver.save(sess, os.path.join('./models/', name, name), global_step=step)
                        print('model checkpoint saved on step {:d}'.format(step))
                    print("size of valid {}, {}".format(len(valid_X), len(valid_y)))
                    if valid_X is not None and valid_y is not None:
                        valid_loss_value1 = sess.run(
                            valid_loss,
                            feed_dict={
                                self.input_text: valid_X,
                                self.output_text: valid_y,
                                self.dropout_rate: 0,
                                self.schedule_sampling_rate: 1 ## TODO: not sure the rate should be
                            })
                        valid_loss_value0 = sess.run(
                            valid_loss,
                            feed_dict={
                                self.input_text: valid_X,
                                self.output_text: valid_y,
                                self.dropout_rate: 0,
                                self.schedule_sampling_rate: 0 ## TODO: not sure the rate should be
                            })
                        print('epoch no.{:d} done, \tvalidation loss0, 1: {:.5f}, {:.5f}'.format(step, np.mean(valid_loss_value0), np.mean(valid_loss_value1)))

                        # Generate sample sentence
                        sample_valid_X = random.sample(list(valid_X), k=5)
                        sample_sentence_gen = sess.run(
                            sample_sentence,
                            feed_dict={
                                self.input_text: sample_valid_X,
                                self.output_text: valid_y,          # NOT IMPORTANT
                                self.dropout_rate: 0,               # NOT IMPORTANT
                                self.schedule_sampling_rate: 0      # NOT IMPORTANT
                            }
                        )
                        for i, x in enumerate(sample_sentence_gen):
                            print('Sample original sentence: ' + ' '.join(Data.get_words_by_indices(sample_valid_X[i])))
                            print('Sample generated sentence: ' + ' '.join(Data.get_words_by_indices(x)))
                    else:
                        print('epoch no.{:d} done.'.format(step))
                    for batch_idx, (batch_X, batch_y) in enumerate(Data.get_next_batch(batch_size, X, y)):
                        # TODO: schedule_sampling
                        feed_dict = {
                            self.input_text: batch_X,
                            self.output_text: batch_y,
                            self.schedule_sampling_rate: schedule_sampling_rate,
                            self.dropout_rate: dropout_rate
                        }

                        # if self.use_dropout:
                        #     feed_dict[self.dropout_rate] = dropout_rate
                        # if self.use_ss:
                        #     if step < 2:
                        #         feed_dict[self.schedule_sampling_rate] = 1.0
                        #     else:
                        #         feed_dict[self.schedule_sampling_rate] = 0.5

                        _, train_loss_value = sess.run(
                            [optimizer, loss],
                            feed_dict=feed_dict
                        )

                        feed_dict = {
                            self.input_text: batch_X,
                            self.output_text: batch_y,
                            self.schedule_sampling_rate: 1,
                            self.dropout_rate: 0
                        }
                        train_loss_value = sess.run(
                            loss,
                            feed_dict=feed_dict
                        )
                        print('epoch no.{:d}, batch no.{:d}, loss: {:.6f}'.format(step, batch_idx, train_loss_value), end='\r', flush=True)



    def restore(self, sess, model_path='./model/'):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

    def predict(self, X, reuse=True, model_path=''):
        """
        Parameters
        ----------
        X               shape (# samples, 80, 4096)
        model_dir       string directory name of models
        name            string, name of model
        model_epoch     integer, specific checkpoint of the model

        Returns
        -------
        y_pred          shape (# samples, beam_size, length of sentences)

        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        _, pred_words = self.build_model(batch_size=len(X), is_training=False)

        pred = sess.run(pred_words, feed_dict={
            self.input_text: X,
            self.dropout_rate: 0
        })

        return pred

    def multi_lstm(self, name, dropout_rate=None):
        print(dropout_rate)
        # with tf.variable_scope(name or 'LSTM'):
        def cell():
            return rnn.BasicLSTMCell(self.n_hidden)

        att_cell = cell
        if dropout_rate != None:
            def att_cell():
                return rnn.DropoutWrapper(cell=cell(), output_keep_prob=(1 - dropout_rate))

        if self.n_layers > 1:
            return rnn.MultiRNNCell([att_cell() for _ in range(self.n_layers)])
        return att_cell()

    def linear(self, input_, output_size, name ,scope=None, stddev=0.02, bias_start=0.0, with_w=False):
        shape = input_.get_shape().as_list()

        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("{}_M".format(name), [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("{}_b".format(name), [output_size], initializer=tf.constant_initializer(bias_start))
            if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias
