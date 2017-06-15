# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, os
import numpy as np
import argparse
import collections
import tensorflow as tf
from random import shuffle, sample
import nltk
from nltk.tokenize import word_tokenize
remove_list = ['.', ',', '?', '"', '!', '/','<i>','</ i', '[', ']', 'i>', '<muslc>', '>', '<', '#', '-','+++$+++']
replace_list = [['`','\''], ['´','\''], ['’','\''], ['\' ', '\'']]

def special_word_to_id(word):
    if word == '<PAD>':
        return 0
    if word == '<UNK>':
        return 1
    if word == '<BOS>':
        return 2
    if word == '<EOS>':
        return 3
    print("%s is not special word!\n" % word)
    return 0

def remove(line, remove_list, replace_list=replace_list):
    for i in remove_list:
        line = line.replace(i, '')
    for i in replace_list:
        line = line.replace(i[0], i[1])
    return line

def _read_words(filenames):
    all_words = []
    for filename in filenames:
        with open(filename, "r", encoding='utf-8', errors='replace') as f:
            all_words += word_tokenize(remove(f.read().lower().replace("\n", ""), remove_list))
            # all_words += word_tokenize(remove(f.read().lower().replace("\n", ""), remove_list)
    return all_words

def _read_lines(filenames):
    lines = []
    for filename in filenames:
        with open(filename, "r", encoding='utf-8', errors='replace') as f:
            for line in f:
                lines.append(word_tokenize(remove(line.lower(), remove_list)))
    return lines

def _build_vocab(filename, vocab_size):
    data = _read_words(filename)

    counter = collections.Counter(data).most_common(vocab_size - 4)
    count_pairs = sorted(counter, key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    words = list(words)
    # print(words)
    words.insert(0, '<PAD>') # 0
    words.insert(1, '<UNK>') # 1
    words.insert(2, '<BOS>') # 2
    words.insert(3, '<EOS>') # 3
    word_to_id = dict(zip(words, range(len(words))))
    # word_to_id['<EOS>'] = -1
    # word_to_id['<BOS>'] = -2
    # word_to_id['<UNK>'] = -3
    # word_to_id['<PAD>'] = -4
    return word_to_id


def _file_to_word_ids(filenames, word_to_id, min_len=4):
    total_unk = 0
    total_words = 0
    def sentence2ids(sentence):
        line = []
        unk_count = 0
        for word in sentence:
            id = word_to_id.get(word, word_to_id['<UNK>']) # <UNK>
            if id == word_to_id['<UNK>']:
                unk_count += 1
            line += [id]
        return line, unk_count
    data = _read_lines(filenames)
    i = 0
    pairs = []
    while i + 1 < len(data):
        line1, line1_unk = sentence2ids(data[i])
        line2, line2_unk = sentence2ids(data[i+1])
        # line1 = [word_to_id[word] for word in data[i] if word in word_to_id]
        # line2 = [word_to_id[word] for word in data[i+1] if word in word_to_id]
        if len(line1) > 0 and len(line2) > min_len and max(line1_unk, line2_unk) < 3:
            total_unk += (line1_unk + line2_unk)
            total_words += (len(line1), len(line2))
            pairs.append([line1, line2])
        i += 2
    print(total_unk * 1.0 / total_words)
    print(total_unk)
    print(total_words)
    return pairs

def _save_dict(_dict, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for key, val in _dict.items():
            f.write("%s %d\n" % (key, val))

def gen_training_data(data_path=None, save_path=None, dict_path='dict.txt', vocab_size=5000, min_len=4, valid_ratio=0.01, test_ratio=0.05):

    word_to_id = _build_vocab(data_path, vocab_size)
    _save_dict(word_to_id, dict_path)
    train_data = _file_to_word_ids(data_path, word_to_id, min_len)
    print('training')
    # print(train_data)
    shuffle(train_data)
    valid_size = int(len(train_data) * valid_ratio)
    test_size = int(len(train_data) * test_ratio)
    print(valid_size)
    np.save("{}_train.npy".format(save_path), train_data[valid_size+test_size:])
    np.save("{}_valid.npy".format(save_path), train_data[:valid_size])
    np.save("{}_test.npy".format(save_path), train_data[valid_size:valid_size+test_size])
    vocabulary = len(word_to_id)
    return train_data, vocabulary

if __name__ == '__main__':
    # do something
    # '/tmp/chat_corpus/open_subtitles.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', '-c',
                        help='path to corpus file', action='append')
    parser.add_argument('--vocab_size',
                        help='size of vocabulary', default=20000, type=int)
    parser.add_argument('--save_path',
                        help='path to save data')
    parser.add_argument('--dict_path',
                        help='path to save dictionary')
    parser.add_argument('--min_len',
                        help='the min length of each sentence', default=4,type=int)
    parser.add_argument('--valid_ratio',
                        help='the ratio for validation set', default=0.01,type=float)
    args = parser.parse_args()
    print(args.corpus_path)
    train, vocab = gen_training_data(args.corpus_path, args.save_path, args.dict_path, vocab_size=args.vocab_size, min_len=args.min_len, valid_ratio=args.valid_ratio)
    print(vocab)
