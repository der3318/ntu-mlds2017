# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, os
import numpy as np
import argparse
import collections
import tensorflow as tf
remove_list = ['.', ',', '?', '"', '!', '/','<i>','</ i', '[', ']', 'i>', '<muslc>', '>', '<', '#']
replace_list = ['`', '´', '’']

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
        line = line.replace(i, '\'')
    return line

# def read_corpus(file_path):
#     with open(file_path, 'r') as f:
#         for index, line in enumerate(f):
#             line = remove(line, remove_list)
#             line = line.split()
#             if index > 30:
#                 break
#             print(line)


def _read_words(filename):
    with open(filename, "r", encoding='utf-8') as f:
        return remove(f.read().lower().replace("\n", ""), remove_list).split()

def _read_lines(filename):
    lines = []
    with open(filename, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(remove(line.lower(), remove_list).split())
    return lines

def _build_vocab(filename, vocab_size):
    data = _read_words(filename)

    counter = collections.Counter(data).most_common(vocab_size)
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


def _file_to_word_ids(filename, word_to_id, min_len=4):
    data = _read_lines(filename)
    i = 0
    pairs = []
    while i + 1 < len(data):
        line1 = [word_to_id[word] for word in data[i] if word in word_to_id]
        line2 = [word_to_id[word] for word in data[i+1] if word in word_to_id]
        if len(line1) > 0 and len(line2) > min_len:
            pairs.append([line1, line2])
        i += 2
    print(pairs)
    return pairs

def _save_dict(_dict, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for key, val in _dict.items():
            f.write("%s %d\n" % (key, val))
def gen_training_data(data_path=None, save_path=None, vocab_size=5000, min_len=4):

    word_to_id = _build_vocab(data_path, vocab_size)
    _save_dict(word_to_id, 'dict.txt')
    train_data = _file_to_word_ids(data_path, word_to_id, min_len)
    np.save(save_path, train_data)
    vocabulary = len(word_to_id)
    return train_data, vocabulary

if __name__ == '__main__':
    # do something
    # '/tmp/chat_corpus/open_subtitles.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path',
                        help='path to corpus file')
    parser.add_argument('--vocab_size',
                        help='size of vocabulary', default=5000)
    parser.add_argument('--save_path',
                        help='path to save data')
    parser.add_argument('--min_len',
                        help='the min length of each sentence', default=4,type=int)
    args = parser.parse_args()
    train, vocab = gen_training_data(args.corpus_path, args.save_path, min_len=args.min_len)
    print(vocab)
