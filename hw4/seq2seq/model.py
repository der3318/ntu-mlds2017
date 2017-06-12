# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import math, os, sys, random
from preprocessing import special_word_to_id
from tensorflow.contrib import legacy_seq2seq
class seq2seq:
    def __init__(self,
                n_layers    = 4,
                n_hidden    = 512,
                n_step1     = 20,
                n_step2     = 20,
                dim_input   = 20000,
                dim_output  = 20000,
                use_ss      = False,
                embedding_dim   = 130,
                # use_att        = True,
                use_bn      = False,
                use_dropout = True,
                beam_size   = 1,
                alpha_c     = 0.0,
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
        # use_att         boolean, whether to use attention
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
        self.embedding_dim = embedding_dim
        self.use_ss     = use_ss
        # self.use_att    = use_att
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
        #     self.encoder = self.multi_lstm('lstm1', dropout_rate=self.dropout_rate)
        #     self.decoder = self.multi_lstm('lstm2', dropout_rate=self.dropout_rate)
        # else:
        #     self.encoder = self.multi_lstm('lstm1', dropout_rate=None)
        #     self.decoder = self.multi_lstm('lstm2', dropout_rate=None)
        # print(self.encoder)
        # print(self.decoder)

        self.input_embed_W = tf.get_variable('input_embed_W', [dim_input, embedding_dim], tf.float32, initializer=tf.random_normal_initializer())
        # self.input_embed_b = tf.get_variable('input_embed_b', [n_hidden], tf.float32, initializer=tf.constant_initializer(0.0))
        self.output_embed_W = tf.get_variable('output_embed_W', [dim_output, embedding_dim], tf.float32, initializer=tf.random_normal_initializer())

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


    def build_model(self, batch_size=128, is_training=True):


        def basic_cell():
            return rnn.BasicLSTMCell(self.n_hidden)

        att_cell = basic_cell
        if self.use_dropout != None:
            def att_cell():
                return rnn.DropoutWrapper(cell=basic_cell(), output_keep_prob=(1 - self.dropout_rate))
        if self.n_layers > 1:
            cell = rnn.MultiRNNCell([att_cell() for _ in range(self.n_layers)])
        else:
            cell = att_cell()

        # cell = rnn.MultiRNNCell([rnn.DropoutWrapper(cell=rnn.BasicLSTMCell(self.n_hidden), output_keep_prob=(1 - self.dropout_rate)) for _ in range(self.n_layers)])
        # cell = rnn.BasicLSTMCell(self.n_hidden)

        encoder_input = tf.unstack(self.input_text, axis=1)
        decoder_input = tf.unstack(self.output_text, axis=1)
        outputs, state = legacy_seq2seq.embedding_attention_seq2seq(
            encoder_input,
            decoder_input,
            cell,
            num_encoder_symbols=self.dim_input,
            num_decoder_symbols=self.dim_output,
            embedding_size=self.embedding_dim,
            num_heads=5,
            output_projection=(self.output_proj_W, self.output_proj_b),
            feed_previous=self._sample(tf.constant(False), tf.constant(True), self.schedule_sampling_rate),
            initial_state_attention=False
        )
        # outputs: A list of the same length as decoder_inputs of 2D Tensors with shape [batch_size x num_decoder_symbols]

        outputs = [tf.nn.xw_plus_b(output, self.output_proj_W, self.output_proj_b) for output in outputs]

        if is_training:
            loss = legacy_seq2seq.sequence_loss_by_example(
                logits=outputs[:-1],
                targets=decoder_input[1:],
                weights=[tf.ones([batch_size]) for _ in range(self.n_step2 - 1)],
                average_across_timesteps=True,
                softmax_loss_function=None,
                name=None
            )
            loss = tf.reduce_sum(loss) / batch_size

            reshape_output = tf.nn.softmax(tf.transpose(tf.stack(outputs), perm=[1,0,2])) + np.finfo(np.float32).eps # [batch_size, n_steps, dim_output]
            perplexity = 0.0
            for i in range(batch_size):
                entropy = - tf.reduce_sum( tf.multiply(reshape_output[i,:,:], tf.log(reshape_output[i,:,:]) ) )  / (self.n_step2 - 1)
                # entropy =  - tf.reduce_sum(tf.log(tf.nn.softmax(reshape_output[i,:,:]), axis=1)) / (self.n_step2 - 1)
                perplexity += tf.exp(entropy)
            perplexity /= batch_size

            return loss, outputs, perplexity


        mask = [1] * self.dim_output
        mask[special_word_to_id('<UNK>')] = 0
        mask = [mask] * batch_size
        outputs = [tf.nn.softmax(output) * mask for output in outputs]
        return tf.argmax(tf.transpose(tf.stack(outputs), perm=[1,0,2]), axis=2)

    def train(self, Data, batch_size=64, learning_rate=1e-3, epoch=100, period=3, name='model', dropout_rate=0.0, resume_model_path=None, start_step=0, ss_rate=1.0):
        """
        Parameters
        ----------
        Data            instance of constructed data class
        batch_size      integer, number of samples in a batch
        learning_rate   float
        epoch           integer, number of epochs to run
        period          integer, intervals between checkpoints
        name            string, model name to save
        dropout_rate    float
        resume_model_path   the path of resume path, otherwise, None
        ss_rate
        """
        # TODO: 如果是restore就不intialize

        # X, y, valid_X, valid_y = Data.gen_train_data(test_ratio=0.01)
        valid_X, valid_y = Data.gen_valid_data()
        print('Training')
        os.makedirs(os.path.join('./models/', name), exist_ok=True)
        with tf.variable_scope('Model') as scope:
            loss, _, perplexity = self.build_model(batch_size=batch_size, is_training=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            print('Model built with batch size = {}'.format(batch_size))
            scope.reuse_variables()
            valid_loss, valid_output, _ = self.build_model(batch_size=batch_size, is_training=True)
            sample_sentence = self.build_model(batch_size=5, is_training=False)

            # sample_rates = [1, 0.9]
            # sample_rate_boundary = [10, epoch * 3 / 4, epoch]
            # TODO: gradient clipping

            saver = tf.train.Saver(max_to_keep=3)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                if resume_model_path != None:
                    saver.restore(sess, resume_model_path)
                    print('Restore model done')

                for step in range(start_step, epoch):

                    # for i in range(len(sample_rates)):
                    #     if step <= sample_rate_boundary[i]:
                    #         schedule_sampling_rate = sample_rates[i]
                    #         break
                    schedule_sampling_rate = ss_rate
                    print("The schedule sampling rate is {}".format(schedule_sampling_rate))
                    # TODO: valid_X is not the size of (batch_size * n_hidden), so will cause ERROR

                        # saver.save(sess, os.path.join('./models/', name, name), global_step=step)
                    print("size of valid {}, {}".format(len(valid_X), len(valid_y)))
                    if valid_X is not None and valid_y is not None:
                        valid_loss_value0, valid_loss_value1 = 0.0, 0.0
                        valid_iter = 0
                        for valid_idx in range(0, len(valid_X) - batch_size + 1, batch_size):
                            valid_iter += 1
                            print("valid total iter no. {}".format(valid_iter), end='\r', flush=True)
                            valid_loss_value1 += np.mean(sess.run(
                                valid_loss,
                                feed_dict={
                                    self.input_text: valid_X[valid_idx:valid_idx+batch_size],
                                    self.output_text: valid_y[valid_idx:valid_idx+batch_size],
                                    self.dropout_rate: 0,
                                    self.schedule_sampling_rate: 1
                                }))
                            valid_loss_value0 += np.mean(sess.run(
                                valid_loss,
                                feed_dict={
                                    self.input_text: valid_X[valid_idx:valid_idx+batch_size],
                                    self.output_text: valid_y[valid_idx:valid_idx+batch_size],
                                    self.dropout_rate: 0,
                                    self.schedule_sampling_rate: 0
                                }))

                        print('epoch no.{:d} done, \tvalidation loss0, 1: {:.5f}, {:.5f}'.format(step, valid_loss_value0 / valid_iter, valid_loss_value1 / valid_iter))

                        # Generate sample sentence
                        sample_valid_X = random.sample(list(valid_X), k=5)
                        sample_sentence_gen = sess.run(
                            sample_sentence,
                            feed_dict={
                                self.input_text: sample_valid_X,
                                self.output_text: sample_valid_X,   # NOT IMPORTANT
                                self.dropout_rate: 0,               # NOT IMPORTANT
                                self.schedule_sampling_rate: 0
                            }
                        )
                        for i, x in enumerate(sample_sentence_gen):
                            filtered_x = []
                            for _ in x:
                                if _ == 3: break
                                filtered_x += [_]
                            print('o: ' + ' '.join(Data.get_words_by_indices(sample_valid_X[i])))
                            print('g: ' + ' '.join(Data.get_words_by_indices(filtered_x)))
                    else:
                        print('epoch no.{:d} done.'.format(step))
                    for batch_idx, (batch_X, batch_y) in enumerate(Data.get_next_batch(batch_size)):
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

                        _, train_loss_value, perplex_val = sess.run(
                            [optimizer, loss, perplexity],
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
                        print('epoch no.{:d}, batch no.{:d}, loss: {:.6f}, perplexity: {:.4f}'.format(step, batch_idx, train_loss_value, perplex_val), end='\r', flush=True)

                    if step % period == 0:
                        save_path = saver.save(sess, "./models/{}/model_after_epoch_{}.ckpt".format(name,step))
                        print('model checkpoint saved on step {:d}'.format(step))


    def restore(self, sess, model_path='./model/'):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

    def build_predict_model(self, batch_size=1, reuse=False):
        """
        Parameters
        ----------
        X               shape (# samples, 80, 4096)
        model_path      string directory name of models
        Data            The data object

        Returns
        -------
        y_pred          shape (# samples, beam_size, length of sentences)

        """
        with tf.variable_scope('Model') as scope:

            if reuse:
                tf.get_variable_scope().reuse_variables()
            _, pred_words = self.build_model(batch_size=batch_size, is_training=False)

        return pred_words

    def predict(self, X, sess, Data, predict_tensor):

        pred = sess.run(predict_tensor, feed_dict={
            self.input_text: X,
            self.output_text: X,             # NOT IMPORTANT
            self.dropout_rate: 0,               # NOT IMPORTANT
            self.schedule_sampling_rate: 0      # NOT IMPORTANT
        })

        return pred
