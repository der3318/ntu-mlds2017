import numpy as np
import json
import csv
import sys
import os

from random import shuffle, sample

from utility import *


class data:
    def __init__(self, testing_dir='MLDS_hw2_data/testing_data/feat/', 
                       testing_id_file='MLDS_hw2_data/testing_id.txt',
                       word_count_threshold=0):
        
        self.testing_data_dictionary,self.testing_data_list = self.load_data(testing_dir,testing_id_file)
        self.word_count_threshold = word_count_threshold
        self.load()


    def load_data(self, data_dir, data_id_file):
        data_dictionary = {}
        data_list = [str(row.strip()) for row in open(data_id_file,"r")]

        for id in data_list:
            data_dictionary[id] = {"data":np.load(os.path.join(data_dir,str(id)+".npy"))}
        return data_dictionary, data_list


    def gen_validation_data(self, cross=False, caption_num=1, best=True):
        shuffle(self.training_data_list)
        if cross:
            fold_size = int(len(self.training_data_list)/5)
            train_X = [[] for i in range(5)]
            train_y = [[] for i in range(5)]
            valid_X = [[] for i in range(5)]
            valid_y = [[] for i in range(5)]
            for fold in range(5):
                for id_index in range(len(self.training_data_list)):
                    id = self.training_data_list[id_index]
                    if id_index in range(fold*fold_size,(fold+1)*fold_size):
                        valid_X[fold] += [self.training_data_dictionary[id]["data"]]
                        valid_y[fold] += [self.training_data_dictionary[id]["caption"]]
                    else:
                        if caption_num > len(self.training_data_dictionary[id]["caption"]):
                            train_X[fold] += [id]*len(self.training_data_dictionary[id]["caption"])
                            train_y[fold] += self.training_data_dictionary[id]["caption"]
                        else:
                            train_X[fold] += [id]*caption_num
                            if best:
                                train_y[fold] += get_best_k_caption(self.training_data_dictionary[id]["caption"],caption_num)
                            else:
                                train_y[fold] += sample(self.training_data_dictionary[id]["caption"],caption_num)
            return train_X, train_y, np.asarray(valid_X), np.asarray(valid_y)

        else:
            index = int(len(self.training_data_list)*4/5)
            valid_X = [self.training_data_dictionary[id]["data"] for id in self.training_data_list[index:]]
            valid_y = [self.training_data_dictionary[id]["caption"] for id in self.training_data_list[index:]]
            train_X = []
            train_y = []
            for id in self.training_data_list[:index]:
                if caption_num > len(self.training_data_dictionary[id]["caption"]):
                    train_X += [id]*len(self.training_data_dictionary[id]["caption"])
                    train_y += self.training_data_dictionary[id]["caption"]
                else:
                    train_X += [id]*caption_num
                    if best:
                        train_y += self.get_best_k_caption(self.training_data_dictionary[id]["caption"],caption_num)
                    else:
                        train_y += sample(self.training_data_dictionary[id]["caption"],caption_num)
            return train_X, train_y, np.asarray(valid_X), np.asarray(valid_y)



    def get_best_k_caption(self,captions,caption_num=1):
        import bleu
        captions = list(map(str,captions))
        score_list = [[bleu.eval(captions[y_index],captions[:y_index]+captions[y_index+1:]),y_index] for y_index in range(len(captions))]
        score_list = sorted(score_list, key = lambda x : x[0], reverse=True)
        selected_list = [captions[score_list[k][1]] for k in range(caption_num)]
        return selected_list

    
    
    def get_vocab_size(self):
        return len(self.wordtoix)



    def get_index_by_word(self, word):
        return self.wordtoix.get(word, 0)



    def get_word_by_index(self, index):
        return self.ixtoword.get(index, '<unk>')



    def get_words_by_indices(self, indices):
        strings = [self.get_word_by_index(index) for index in indices]
        words = [word for word in strings if word not in ['<bos>', '<eos>', '<pad>', '<unk>']]
        return words



    def get_sentence_by_indices(self, indices):
        return ' '.join(self.get_words_by_indices(indices))



    def gen_test_data(self):
        test_X = [self.testing_data_dictionary[id]["data"] for id in self.testing_data_list]
        test_id = [id for id in self.testing_data_list]

        return np.asarray(test_X), np.asarray(test_id)



    def gen_train_data(self, caption_num=1, best=True):
        shuffle(self.training_data_list)
        
        test_X, test_y = self.gen_test_data()

        train_X = []
        train_y = []
        for id in self.training_data_list:
            if caption_num > len(self.training_data_dictionary[id]["caption"]):
                train_X += [id]*len(self.training_data_dictionary[id]["caption"])
                train_y += self.training_data_dictionary[id]["caption"]
            else:
                train_X += [id]*caption_num
                if best:
                    train_y += self.get_best_k_caption(self.training_data_dictionary[id]["caption"],caption_num)
                else:
                    train_y += sample(self.training_data_dictionary[id]["caption"],caption_num)
        return train_X, train_y, test_X, test_y



    def process_sentence(self, sentence, n_step, comma=False):
        for char in [',', '"', '?', '!', '\\', '/']:
            sentence = sentence.replace(char,"") if comma == False else sentence.replace(char,char+" ")

        sentence = ['<bos>'] + [word for word in sentence.strip().split('.')[0].split(' ') if word.lower() in self.wordtoix]
        if len(sentence) > n_step-1:
            sentence = sentence[:n_step-1]

        sentence += ['<eos>']
        sentence += ['<pad>'] * (n_step - len(sentence))

        return [self.wordtoix[word.lower()] for word in sentence if word.lower() in self.wordtoix]



    def get_next_batch(self, batch_size, train_X, train_y, n_step):
        train_data = list(zip(train_X,train_y))
        shuffle(train_data)
        train_X, train_y = zip(*train_data)

        for offset in range(0, len(train_X)-batch_size+1 ,batch_size):
            batch_X = np.asarray([self.training_data_dictionary[id]["data"] for id in train_X[offset:offset+batch_size]])
            batch_y = np.asarray([self.process_sentence(sentence,n_step) for sentence in train_y[offset:offset+batch_size]])
            yield batch_X, batch_y



    def save(self, dir_path='./data/'):
        path_wordtoix = os.path.join(dir_path, 'wordtoix-'+str(self.word_count_threshold))
        path_ixtoword = os.path.join(dir_path, 'ixtoword-'+str(self.word_count_threshold))
        
        np.save(path_wordtoix, self.wordtoix)
        np.save(path_ixtoword, self.ixtoword)


    
    def load(self, dir_path='./data/'):
        path_wordtoix = os.path.join(dir_path, 'wordtoix-'+str(self.word_count_threshold)+'.npy')
        path_ixtoword = os.path.join(dir_path, 'ixtoword-'+str(self.word_count_threshold)+'.npy')

        if not os.path.exists(path_wordtoix) or not os.path.exists(path_ixtoword):
            return False

        self.wordtoix = np.load(path_wordtoix).tolist()
        self.ixtoword = np.load(path_ixtoword).tolist()
        
        return True
