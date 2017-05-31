# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
from random import shuffle, sample
from preprocessing import special_word_to_id
class data:
    def __init__(self, train_data_path, dict_path, n_step=20):

        (self.word2id, self.id2word) = self.load_dict(dict_path)
        self.train_data = self.load_data(train_data_path)
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

    def get_word_by_index(self, index):
        return self.id2word.get(index, '<UNK>')

    def get_words_by_indices(self, indices):
        strings = [self.get_word_by_index(index) for index in indices]
        words = [word for word in strings if word not in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']]
        return words

    def get_sentence_by_indices(self, indices):
        return ' '.join(self.get_words_by_indices(indices))


    # def gen_test_data(self):
    #     test_X = [self.testing_data_dictionary[id]["data"] for id in self.testing_data_list]
    #     test_y = [self.testing_data_dictionary[id]["caption"] for id in self.testing_data_list]
    #
    #     return np.asarray(test_X), np.asarray(test_y)

    def process_sentence(self, id_list):
        bos = [self.get_index_by_word('<BOS>')]
        eos = [self.get_index_by_word('<EOS>')]
        pad_id = self.get_index_by_word('<PAD>')
        return bos + id_list + eos + ([pad_id] * (self.n_step - len(id_list) - 2) )

    def gen_train_data(self, test_ratio=0.05):
        shuffle(self.train_data)

        test_X, test_y = [], []
        train_X = []
        train_y = []
        testing_size = len(self.train_data) * test_ratio

        for idx, id in enumerate(self.train_data):
            if len(id[0]) > self.n_step - 2 or len(id[1]) > self.n_step - 2:
                continue
            if idx >= testing_size:
                train_X += [id[0]]
                train_y += [id[1]]
            else:
                test_X += [id[0]]
                test_y += [id[1]]

        test_X = np.asarray([self.process_sentence(sentence) for sentence in test_X])
        test_y = np.asarray([self.process_sentence(sentence) for sentence in test_y])
        return train_X, train_y, test_X, test_y


    def get_next_batch(self, batch_size, train_X, train_y):
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
