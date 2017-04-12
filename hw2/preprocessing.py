import numpy as np
import json
import csv
import sys
import os

from utility import *
from random import (
	shuffle,
	sample,
	)

class data:
	def __init__(self, training_dir='MLDS_hw2_data/training_data/feat/', 
					   training_label_file='MLDS_hw2_data/training_label.json', 
					   testing_dir='MLDS_hw2_data/testing_data/feat/', 
					   testing_label_file='MLDS_hw2_data/testing_public_label.json'):
		self.training_data_dictionary,self.training_data_list = self.load_data(training_dir,training_label_file)
		self.testing_data_dictionary,self.testing_data_list = self.load_data(testing_dir,testing_label_file)
		all_captions = []
		for id in self.training_data_dictionary:
			all_captions += self.training_data_dictionary[id]['caption']
		self.wordtoix, self.ixtoword = preProBuildWordVocab(all_captions, word_count_threshold=0)


	def load_data(self, data_dir, data_label_file):
		data_dictionary = {}
		data_list = []
		with open(data_label_file) as label_file:
			label = json.load(label_file)
		for item in label:
			data_dictionary[item["id"]] = {"data":np.load(data_dir+item["id"]+".npy")}
			data_dictionary[item["id"]]["caption"] = item["caption"]
			data_list.append(item["id"])
		return data_dictionary,data_list


	def gen_validation_data(self, cross=False, caption_num=1):
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
							train_y[fold] += sample(self.training_data_dictionary[id]["caption"],caption_num)
			return train_X,train_y,valid_X,valid_y

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
					train_y += sample(self.training_data_dictionary[id]["caption"],caption_num)
			return train_X,train_y,valid_X,valid_y


	def gen_train_data(self, caption_num=1):
		shuffle(self.training_data_list)
		test_X = [self.testing_data_dictionary[id]["data"] for id in self.testing_data_list]
		test_y = [self.testing_data_dictionary[id]["caption"] for id in self.testing_data_list]
		train_X = []
		train_y = []
		for id in self.training_data_list:
			if caption_num > len(self.training_data_dictionary[id]["caption"]):
				train_X += [id]*len(self.training_data_dictionary[id]["caption"])
				train_y += self.training_data_dictionary[id]["caption"]
			else:
				train_X += [id]*caption_num
				train_y += sample(self.training_data_dictionary[id]["caption"],caption_num)
		return train_X,train_y,test_X,test_y

	def process_sentence(self,sentence,n_step,comma=False):
		for char in [',', '"', '?', '!', '\\', '/']:
			sentence = sentence.replace(char,"") if comma == False else sentence.replace(char,char+" ")
		sentence = ['<bos>'] + sentence.strip().split('.')[0].split(' ') + ['<eos>']
		sentence += ['<pad>'] * (n_step - len(sentence))
		return [self.wordtoix[word.lower()] for word in sentence]

	def get_next_batch(self, batch_size, train_X, train_y, n_step):
		train_data = list(zip(train_X,train_y))
		shuffle(train_data)
		train_X,train_y = zip(*train_data)
		for offset in range(0, len(train_X),batch_size):
			batch_X = np.asarray([self.training_data_dictionary[id]["data"] for id in train_X[offset:offset+batch_size]])
			batch_y = np.asarray([self.process_sentence(sentence,n_step) for sentence in train_y[offset:offset+batch_size]])
			yield batch_X, batch_y

