import tensorflow as tf
import numpy as np
import os
import sys
import math
import random

import bleu

from tensorflow.contrib import rnn


class S2VTmodel:

    def __init__(self, n_hidden=256, n_step1=80, n_step2=20, n_words=3000, dim_image=4096, seed=3318):
        """
        Parameters
        ----------
        n_hidden        integer, number of hidden units
        n_step1         integer, n_steps of the encoding LSTM
        n_step2         integer, n_steps of the decoding LSTM
        n_words         integer, vocabulary size
        dim_image       integer, dimension of input image
        seed            integer, ramdom seed
        
        """
        
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        self.n_hidden   = n_hidden
        self.n_step1    = n_step1
        self.n_step2    = n_step2
        self.n_words    = n_words
        self.dim_image  = dim_image
        self.seed       = seed
        
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, n_hidden], -0.1, 0.1), name='Wemb')
        
        self.lstm1 = rnn.BasicLSTMCell(n_hidden, state_is_tuple=False)
        self.lstm2 = rnn.BasicLSTMCell(n_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, n_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([n_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([n_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')
                  
        self.video = tf.placeholder(tf.float32, [None, self.n_step1, self.dim_image])
        self.caption = tf.placeholder(tf.int32, [None, self.n_step2+1])
        #self.caption_mask = tf.placeholder(tf.float32, [None, self.n_steps2+1])
        

    def build_model(self, batch_size=50):

        video = self.video
        caption = self.caption

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [-1, self.n_step1, self.n_hidden])
       
        state1 = tf.zeros([batch_size, self.lstm1.state_size])
        state2 = tf.zeros([batch_size, self.lstm2.state_size])
        padding = tf.zeros([batch_size, self.n_hidden])

        generated_words = []
        probs = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_step1):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], axis=1), state2)

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_step2): ## Phase 2 => only generate captions
            
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], axis=1), state2)

            labels = tf.expand_dims(caption[:, i+1], axis=1)
            indices = tf.expand_dims(tf.range(0, batch_size, 1), axis=1)
            concated = tf.concat([indices, labels], axis=1)
            onehot_labels = tf.sparse_to_dense(concated, np.asarray([batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
            #cross_entropy = cross_entropy * caption_mask[:,i]
            max_prob_index = tf.argmax(logit_words, axis=1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy) / batch_size
            loss = loss + current_loss

        return loss


    def build_generator(self, batch_size=50):

        video = self.video
        caption = self.caption

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [-1, self.n_step1, self.n_hidden])
       
        state1 = tf.zeros([batch_size, self.lstm1.state_size])
        state2 = tf.zeros([batch_size, self.lstm2.state_size])
        padding = tf.zeros([batch_size, self.n_hidden])

        generated_words = []
        probs = []
        embeds = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_step1):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], axis=1), state2)

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_step2):
            if i == 0:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([batch_size], dtype=tf.int64))

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], axis=1), state2)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, axis=1)
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device('/cpu:0'):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

            embeds.append(current_embed)

        generated_words = tf.transpose(generated_words, perm=[1, 0])

        return generated_words


    def train(self, Data, X, y, valid_X=None, valid_y=None, batch_size=50, learning_rate=1e-3, epoch=2000, period=100, name='model'):
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
        
        """

    
        loss = self.build_model(batch_size=batch_size)
        tf.get_variable_scope().reuse_variables()
        generated_words = self.build_generator(batch_size=len(valid_X))
        
        with tf.variable_scope(tf.get_variable_scope(),reuse=False):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            saver = tf.train.Saver(max_to_keep=10)            
            init = tf.global_variables_initializer()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            for step in range(epoch):
                
                for b, (batch_X, batch_y) in enumerate(Data.get_next_batch(batch_size, X, y, self.n_step2+1)):
                    _, loss_val = sess.run(
                                    [optimizer, loss], 
                                    feed_dict={
                                        self.video: batch_X,
                                        self.caption: batch_y
                                    })

                    print('epoch no.{:d}, batch no.{:d}, loss: {:.6f}'.format(step, b, loss_val))
               
                pred = sess.run(
                        generated_words,
                        feed_dict={
                            self.video: valid_X,
                        })

                scores = []
                for v, (p, vy) in enumerate(zip(pred, valid_y)):
                    sent = Data.get_sentence_by_indices(p)

                    scores.append(bleu.eval(sent, vy))
               
                print('epoch no.{:d} done, \tvalidation score: {:.5f}'.format(step, np.mean(scores)))

                if step % period == period-1:
                    saver.save(sess, os.path.join('models/', name), global_step=step)
                    print('model checkpoint saved on step {:d}'.format(step))

        
    
    def predict(self, X, model_path='./models/'):
        """
        Parameters
        ----------
        X               shape (# samples, 80, 4096)
        model_path      string, path to test model
        
        Returns
        -------
        y_pred          shape (# samples, length of sentences)
        
        """
        generated_words = self.build_generator(batch_size=len(X))
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            saver = tf.train.Saver()
            #save_path = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, model_path)

            pred = sess.run(
                    generated_words,
                    feed_dict={self.video: X}
                    )

        return pred
