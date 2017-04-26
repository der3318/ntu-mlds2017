import tensorflow as tf
import numpy as np
import os
import sys
import math
import random

import bleu

from tensorflow.contrib import rnn


class S2VTmodel:

    def __init__(self, n_hidden=256, n_step1=80, n_step2=20, use_ss=False, use_att=False, use_bn=False, beam_size=1, n_words=3000, dim_image=4096, alpha_c=0.0, seed=3318):
        """
        Parameters
        ----------
        n_hidden        integer, number of hidden units
        n_step1         integer, n_steps of the encoding LSTM
        n_step2         integer, n_steps of the decoding LSTM
        use_ss          boolean, whether to use schedule sampling
        use_att         boolean, whether to use attention
        use_bn          boolean, whether to use batch normalization
        beam_size       integer, number of beams
        n_words         integer, vocabulary size
        dim_image       integer, dimension of input image
        alpha_c         integer, regularization parameters for attention
        seed            integer, ramdom seed
        
        """
        
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        self.n_hidden   = n_hidden
        self.n_step1    = n_step1
        self.n_step2    = n_step2
        self.use_ss     = use_ss
        self.use_att    = use_att
        self.use_bn     = use_bn
        self.beam_size  = beam_size
        self.n_words    = n_words
        self.dim_image  = dim_image
        self.alpha_c    = alpha_c
        self.seed       = seed
        
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, n_hidden], -0.1, 0.1), name='Wemb')
        
        self.lstm1 = rnn.BasicLSTMCell(n_hidden)
        self.lstm2 = rnn.BasicLSTMCell(n_hidden)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, n_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([n_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([n_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')
                  
        self.video = tf.placeholder(tf.float32, [None, self.n_step1, self.dim_image])
        self.caption = tf.placeholder(tf.int32, [None, self.n_step2+1])
        self.schedule_sampling = tf.placeholder(tf.bool, [self.n_step2])
        


    def _embed_plus_output(self, embed, output):
        w = tf.get_variable('w', [2*self.n_hidden, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [self.n_hidden], initializer=tf.constant_initializer(0.0))

        concated = tf.concat([embed, output], axis=1)
        ht = tf.nn.relu(tf.nn.xw_plus_b(concated, w, b))

        return ht


        
    def _attention_layer(self, outputs, state):
        # outputs:  [batch_size, 80, n_hidden]
        # state:    [batch_size, n_hidden]
       
        w_att = tf.get_variable('w_att', [self.n_hidden, 1], initializer=tf.contrib.layers.xavier_initializer())
        
        h_att = tf.nn.relu(outputs + tf.expand_dims(state, 1))
        out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.n_hidden]), w_att), [-1, self.n_step1])
        alpha = tf.nn.softmax(out_att)  
        context = tf.reduce_sum(outputs * tf.expand_dims(alpha, 2), 1, name='context')

        return context, alpha


    
    def _batch_norm(self, x, mode='train'):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope='batch_norm')



    def _beam_search(self, step, logits, log_beam_probs, beam_path, beam_words):
        probs = tf.log(tf.nn.softmax(logits))

        if step == 0:
            # shape(probs) = [batch_size, 1, n_words]
            probs = tf.reshape(probs, [-1, self.n_words])
        if step > 0:
            # shape(probs) = [batch_size, k, n_words]
            # shape(log_beam_probs[-1]) = [batch_size, k, 1]
            # shape(probs + log_beam_probs[-1]) = [batch_size, k, n_words]
            probs = tf.reshape(probs + log_beam_probs[-1],
                                [-1, self.beam_size * self.n_words])
         
        # step > 0: shape(probs) = [batch_size, k*n_words]
        # step = 0: shape(probs) = [batch_size, n_words]
        best_probs, indices = tf.nn.top_k(probs, k=self.beam_size)

        # shape(best_probs) = shape(indices) = [batch_size, k]
        indices = tf.stop_gradient(indices) # [batch_size, k]
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, self.beam_size, 1])) # [batch_size, k, 1]
        
        # which word in vocabulary
        words = indices % self.n_words
        # which hypothesis it came from
        beam_parent = indices // self.n_words

        beam_words.append(words)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)
        
        beam_parent_kmajor = tf.transpose(beam_parent, perm=[1, 0]) # [k, batch_size]
        words_kmajor = tf.transpose(words, perm=[1, 0])
        
        return beam_parent_kmajor, words_kmajor



    def _get_new_states(self, parents, states):
        # shape(parents) = [k, batch_size]
        # shape(states) = [k, 2, batch_size, n_hidden]
        
        states = tf.transpose(states, perm=[0, 2, 1, 3]) # [k, batch_size, 2, n_hidden]
        
        state_list = []
        for parent in tf.unstack(parents):
            # shape(parent) = [batch_size]
            new_states = [states[idx_beam, idx_in_batch, :, :] for idx_in_batch, idx_beam in enumerate(tf.unstack(parent))] # [batch_size, 2, n_hidden]
            state_list.append(tf.stack(new_states))
        
        # shape(state_list) = [k, batch_size, 2, n_hidden]
        state_list = tf.unstack(tf.transpose(state_list, perm=[0, 2, 1, 3])) # [k, 2, batch_size, n_hidden]

        state_list = [tf.unstack(tensor) for tensor in state_list]
        
        return state_list
        
    
    
    def _get_new_embeds(self, words):
        # shape(words) = [k, batch_size]
        
        embed_list = []
        
        with tf.device("/cpu:0"):
            for word_list in tf.unstack(words):
                embed_list.append(tf.nn.embedding_lookup(self.Wemb, word_list))
        
        return embed_list



    def _gather_beams(self, all_path, all_words):
        # shape(path) = shape(words) = [n_step, batch_size, k]
        all_path = np.asarray(all_path)
        all_words = np.asarray(all_words)
        
        steps = all_words.shape[0]
        n_samples = all_words.shape[1]
        beam_size = all_words.shape[2]
       
        pred = []
        for i in range(n_samples):
            path = all_path[:,i,:]
            words = all_words[:,i,:]

            pred_words = [words[-1]]
            parent = path[-1]
            for step in reversed(range(steps-1)):
                pred_words.append(words[step][parent])
                parent = path[step][parent]

            pred.append(np.transpose(pred_words[::-1], [1, 0]))
            
        return np.asarray(pred)

            

    def build_model(self, batch_size=50):

        video = self.video
        caption = self.caption

        mask = tf.to_float(tf.not_equal(caption, 0))

        if self.use_bn:
            video = self._batch_norm(video, mode='train')

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_step1, n_hidden)
        image_emb = tf.reshape(image_emb, [-1, self.n_step1, self.n_hidden])
       
        c1 = tf.zeros([batch_size, self.n_hidden])
        h1 = tf.zeros([batch_size, self.n_hidden])
        c2 = tf.zeros([batch_size, self.n_hidden])
        h2 = tf.zeros([batch_size, self.n_hidden])
        
        padding = tf.zeros([batch_size, self.n_hidden])

        outputs = []

        generated_words = []
        alpha_list = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_step1):

            with tf.variable_scope("LSTM1", reuse=(i!=0)):
                output1, (c1, h1) = self.lstm1(image_emb[:,i,:], state=[c1, h1])

            outputs.append(output1)

            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, (c2, h2) = self.lstm2(tf.concat([padding, output1], axis=1), state=[c2, h2])

        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, (1, 0, 2))
        
        ############################# Decoding Stage ######################################
        for i in range(0, self.n_step2): ## Phase 2 => only generate captions
            
            if i == 0:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
            else:
                with tf.device("/cpu:0"):
                    current_embed = tf.cond(self.schedule_sampling[i], lambda: tf.nn.embedding_lookup(self.Wemb, caption[:, i]), lambda: tf.nn.embedding_lookup(self.Wemb, max_prob_index))

            with tf.variable_scope("LSTM1", reuse=True):
                output1, (c1, h1) = self.lstm1(padding, state=[c1, h1])

            if self.use_att: 
                with tf.variable_scope("embed_plus_output", reuse=(i!=0)):
                    ht = self._embed_plus_output(current_embed, output1)

                with tf.variable_scope("attention", reuse=(i!=0)):
                    context, alpha = self._attention_layer(outputs, h2)
                    alpha_list.append(alpha)
            else:
                context = output1
                ht = current_embed
            
            with tf.variable_scope("LSTM2", reuse=True):
                output2, (c2, h2) = self.lstm2(tf.concat([ht, context], axis=1), state=[c2, h2])

            labels = tf.expand_dims(caption[:, i+1], axis=1)
            indices = tf.expand_dims(tf.range(0, batch_size, 1), axis=1)
            concated = tf.concat([indices, labels], axis=1)
            onehot_labels = tf.sparse_to_dense(concated, np.asarray([batch_size, self.n_words]), 1.0, 0.0)

            logits = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
            cross_entropy = cross_entropy * mask[:,i+1]
            max_prob_index = tf.argmax(logits, axis=1)
            generated_words.append(max_prob_index)

            current_loss = tf.reduce_sum(cross_entropy) / batch_size
            loss = loss + current_loss

        if self.use_att and self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), perm=[1, 0, 2])
            alphas_all = tf.reduce_sum(alphas, axis=1)
            alpha_reg = self.alpha_c * tf.reduce_sum((16.0/196 - alphas_all) ** 2)
            loss += alpha_reg / batch_size

        return loss


    
    def build_beam_generator(self, batch_size=50):

        video = self.video
        caption = self.caption

        if self.use_bn:
            video = self._batch_norm(video, mode='infer')

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [-1, self.n_step1, self.n_hidden])
       
        c1 = tf.zeros([batch_size, self.n_hidden])
        h1 = tf.zeros([batch_size, self.n_hidden])
        c2 = tf.zeros([batch_size, self.n_hidden])
        h2 = tf.zeros([batch_size, self.n_hidden])
        
        padding = tf.zeros([batch_size, self.n_hidden])

        log_beam_probs = []
        beam_path = []
        beam_words = []
        outputs = []
        generated_words = []

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_step1):
            with tf.variable_scope("LSTM1", reuse=(i!=0)):
                output1, (c1, h1) = self.lstm1(image_emb[:,i,:], state=[c1, h1])

            outputs.append(output1)

            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, (c2, h2) = self.lstm2(tf.concat([padding, output1], axis=1), state=[c2, h2])

        outputs = tf.stack(outputs) # [80, batch_size, n_hidden]
        outputs = tf.transpose(outputs, (1, 0, 2)) # [batch_size, 80, n_hidden]

        ############################# Decoding Stage ######################################
        state_list = [[c2, h2]] # [1, 2, batch_size, n_hidden]
        
        with tf.device("/cpu:0"):
            embed_list = [tf.nn.embedding_lookup(self.Wemb, tf.ones([batch_size], dtype=tf.int64))] # [1, batch_size, n_hidden]
        
        for i in range(0, self.n_step2):
            
            with tf.variable_scope("LSTM1", reuse=True):
                output1, (c1, h1) = self.lstm1(padding, state=[c1, h1])
                # shape(output1) = [batch_size, n_hidden]
                
            logits = []
            states = []
            
            for (c2, h2), embed in zip(state_list, embed_list):
                if self.use_att: 
                    with tf.variable_scope("embed_plus_output", reuse=(i!=0)):
                        ht = self._embed_plus_output(embed, output1)

                    with tf.variable_scope("attention", reuse=(i!=0)):
                        context, _ = self._attention_layer(outputs, h2)
                else:
                    context = output1
                    ht = embed

                with tf.variable_scope("LSTM2", reuse=True):
                    output2, (c2_, h2_) = self.lstm2(tf.concat([ht, context], axis=1), state=[c2, h2])
                
                states.append([c2_, h2_])

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                logits.append(logit_words)
            
            logits = tf.transpose(tf.stack(logits), perm=[1, 0, 2])
            parents, words = self._beam_search(i, logits, log_beam_probs, beam_path, beam_words) # [k, batch_size]
            
            state_list = self._get_new_states(parents, states) # [k, 2, batch_size, n_hidden]
            
            embed_list = self._get_new_embeds(words) # [k, batch_size, n_hidden] 
        
        return beam_path, beam_words


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

        os.makedirs(os.path.join('./models/', name), exist_ok=True)
    
        loss = self.build_model(batch_size=batch_size)
        tf.get_variable_scope().reuse_variables()
        beam_path, beam_words = self.build_beam_generator(batch_size=len(valid_X))
        
        with tf.variable_scope(tf.get_variable_scope(),reuse=False):
            global_step = tf.Variable(0, trainable=False)
            boundaries = [int(epoch*0.5), int(epoch*0.9)]
            values = [learning_rate, learning_rate/3, learning_rate/10]
            lr = tf.train.piecewise_constant(global_step, boundaries, values)

            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            saver = tf.train.Saver(max_to_keep=50)
            init = tf.global_variables_initializer()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            valid_scores = []

            for step in range(epoch):
                for b, (batch_X, batch_y) in enumerate(Data.get_next_batch(batch_size, X, y, self.n_step2+1)):
                    if self.use_ss:
                        sample_probability = float(step)/epoch
                        ss_list = [(np.random.uniform(0, 1) > sample_probability) for i in range(self.n_step2)]
                    else:
                        ss_list = np.ones(self.n_step2).astype(bool)

                    _, loss_val = sess.run(
                                    [optimizer, loss], 
                                    feed_dict={
                                        self.video: batch_X,
                                        self.caption: batch_y,
                                        self.schedule_sampling: ss_list
                                    })

                    print('epoch no.{:d}, batch no.{:d}, loss: {:.6f}'.format(step, b, loss_val), end='\r', flush=True)
                
                if valid_X is not None and valid_y is not None:
                    pred_path, pred_words = sess.run(
                                            [beam_path, beam_words],
                                            feed_dict={
                                                self.video: valid_X
                                            })
        
                    pred = self._gather_beams(pred_path, pred_words)
                    
                    scores = []
                    for v, (p, vy) in enumerate(zip(pred, valid_y)):
                        sent = Data.get_sentence_by_indices(p[0])
                        scores.append(bleu.eval(sent, vy))
                   
                    print('epoch no.{:d} done, \tvalidation score: {:.5f}'.format(step, np.mean(scores)))
                    valid_scores.append(np.mean(scores))
                    

                if step % period == period-1:
                    saver.save(sess, os.path.join('./models/', name, name), global_step=step)
                    print('model checkpoint saved on step {:d}'.format(step))

            np.save(os.path.join('results/', name), np.asarray(valid_scores))    
        
    
    def predict(self, X, model_dir='./models/', name='model', model_epoch=None):
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
        beam_path, beam_words = self.build_beam_generator(batch_size=len(X))
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            saver = tf.train.Saver()
            if model_epoch is None:
                save_path = tf.train.latest_checkpoint(os.path.join(model_dir, name))
            else:
                save_path = os.path.join(model_dir, name, name+'-'+str(model_epoch))

            saver.restore(sess, save_path)
            
            pred_path, pred_words = sess.run(
                                    [beam_path, beam_words],
                                    feed_dict={self.video: X}
                                    )
            """
            pred = sess.run(
                    generated_words,
                    feed_dict={self.video: X}
                    )
            """
        pred = self._gather_beams(pred_path, pred_words)

        return pred

