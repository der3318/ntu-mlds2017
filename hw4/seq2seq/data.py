# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
from random import shuffle, sample
from preprocessing import special_word_to_id
class data:
    def __init__(self, train_data_path, valid_data_path, dict_path, n_step=20):

        (self.word2id, self.id2word) = self.load_dict(dict_path)
        self.train_data = self.load_data(train_data_path)
        self.valid_data = self.load_data(valid_data_path)
        self.n_step = n_step

    def load_dict(self, data_path):
        word2id = dict()
        id2word = dict()
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, id = line.split()
                id = int(id)
                word2id[word] = id
                id2word[id] = word
        return word2id, id2word

    def load_data(self, data_path):
        return np.load(data_path)

    def get_vocab_size(self):
        return len(self.word2id)

    def get_index_by_word(self, word):
        return self.word2id.get(word, special_word_to_id('<UNK>'))

    def get_indices_by_sentence(self, sentence):
        return [self.word2id[word] for word in sentence if word in self.word2id]

    def get_word_by_index(self, index):
        return self.id2word.get(index, '<UNK>')

    def get_words_by_indices(self, indices):
        strings = [self.get_word_by_index(index) for index in indices]
        words = []
        for word in strings:
            if word == '<EOS>':
                break
            if word not in ['<BOS>']: # take out <PAD> for visualization
                words += [word]
        return words
        # words = [word for word in strings if word not in ['<BOS>', '<EOS>', '<PAD>']]
        # return words

    def get_sentence_by_indices(self, indices):
        return ' '.join(self.get_words_by_indices(indices))

    def process_sentence(self, id_list):
        bos = [self.get_index_by_word('<BOS>')]
        eos = [self.get_index_by_word('<EOS>')]
        pad_id = self.get_index_by_word('<PAD>')
        return bos + id_list + eos + ([pad_id] * (self.n_step - len(id_list) - 2) )

    def gen_data(self, data):
        X, y = [], []
        for idx, pair in enumerate(data):
            if len(pair[0]) > self.n_step - 2 or len(pair[1]) > self.n_step - 2:
                continue
            X += [pair[0]]
            y += [pair[1]]
        return X, y

    def gen_train_data(self):
        shuffle(self.train_data)
        train_X, train_y = self.gen_data(self.train_data)

        return train_X, train_y

    def gen_valid_data(self, size=None):
        shuffle(self.valid_data)
        if size != None and size <= len(self.valid_data):
            valid_X, valid_y = self.gen_data(self.valid_data[:size])
        else:
            valid_X, valid_y = self.gen_data(self.valid_data)
        valid_X = np.asarray([self.process_sentence(sentence) for sentence in valid_X])
        valid_y = np.asarray([self.process_sentence(sentence) for sentence in valid_y])

        return valid_X, valid_y

    def get_next_batch(self, batch_size):
        train_X, train_y = self.gen_train_data()
        train_data = list(zip(train_X,train_y))
        shuffle(train_data)
        train_X, train_y = zip(*train_data)

        for offset in range(0, len(train_X)-batch_size+1, batch_size):
            batch_X = np.asarray([self.process_sentence(sentence) for sentence in train_X[offset:offset+batch_size]])
            batch_y = np.asarray([self.process_sentence(sentence) for sentence in train_y[offset:offset+batch_size]])
            yield batch_X, batch_y

    #
    #
    # def save(self, dir_path='./data/'):
    #     path_wordtoix = os.path.join(dir_path, 'wordtoix-'+str(self.word_count_threshold))
    #     path_ixtoword = os.path.join(dir_path, 'ixtoword-'+str(self.word_count_threshold))
    #
    #     np.save(path_wordtoix, self.wordtoix)
    #     np.save(path_ixtoword, self.ixtoword)



    # def load(self, dir_path='./data/'):
    #     path_wordtoix = os.path.join(dir_path, 'wordtoix-'+str(self.word_count_threshold)+'.npy')
    #     path_ixtoword = os.path.join(dir_path, 'ixtoword-'+str(self.word_count_threshold)+'.npy')
    #
    #     if not os.path.exists(path_wordtoix) or not os.path.exists(path_ixtoword):
    #         return False
    #
    #     self.wordtoix = np.load(path_wordtoix).tolist()
    #     self.ixtoword = np.load(path_ixtoword).tolist()
    #
    #     return True
