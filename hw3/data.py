from __future__ import print_function

import os
import sys
import time
import numpy as np

from scipy.spatial.distance import cosine
from keras.preprocessing.image import ImageDataGenerator
from utility import read_tags, read_test_texts
from embedding import Embedding


class Data:
    
    support_noise_type = ['normal', 'uniform']
    
    def __init__(self, train_file, test_file,
                 train_embed_path='./data/embed_skipthoughts.npy',
                 test_embed_path='./data/test',
                 test_only=False,
                 seed=3318):
        """
        Args:
            train_file: path to tag_clean.csv
            test_file: path to testing text file
            train_embed_path: path to training embeddings
            test_embed_path: path to directory contains
                             testing embeddings
            seed: seed for reproducing

        """
        self.train_file = train_file
        self.test_file = test_file
        self.train_embed_path = train_embed_path
        self.test_embed_path = test_embed_path
        self.seed = seed
        self.fixed_noise = None
        self.data_gen = ImageDataGenerator(
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            horizontal_flip=True,
                            data_format='channels_last')
        
        np.random.seed(self.seed)
        
        self.test_texts = read_test_texts(test_file)
        self.embedding = Embedding(train_embed_path, test_embed_path)
        self.train_embeds, self.test_embeds = self.embedding.get_embeds()
        
        if not test_only:
            self.train_tags = read_tags(train_file, min_count=1)
            self._get_images()

            # remove those data with all-zero embedding
            idx = np.sum(self.train_embeds, axis=1) > 0
            idx = np.logical_and(idx, np.sum(self.train_embeds, axis=1) < 4)
            idx = np.arange(self.train_embeds.shape[0])[idx]
            self.images = self.images[idx]
            self.train_embeds = self.train_embeds[idx]
            self.train_tags = [self.train_tags[i] for i in idx]


    def _get_images(self):
        print('loading images... ', end='')
        self.images = np.load('./data/images.npy', mmap_mode='r')
        print('done')
        
        return
        
    def _get_noise(self, noise_type, shape):
        if not noise_type in self.support_noise_type:
            noise_type = 'normal'
        
        if noise_type == 'normal':
            noise = np.random.normal(0, 1, shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-1, 1, shape)
        else:
            noise = np.random.normal(0, 1, shape)
            
        return noise
        
    def __get_wrong_embeds(self, index):
        embed = self.train_embeds[index]
        n_embeds = self.train_embeds.shape[0]
        randidx = np.random.randint(n_embeds)
        
        tag_set = set(self.train_tags[index])
        while len(tag_set.intersection(set(self.train_tags[randidx]))) > 0:
            randidx = np.random.randint(n_embeds)
        
        return self.train_embeds[randidx]
        
    def _get_wrong_embeds(self, indices):
        return np.array([self.__get_wrong_embeds(i) for i in indices]) 
    
    def get_data_length(self):
        return self.train_embeds.shape[0]

    def get_embed_dim(self):
        return self.train_embeds.shape[1]

    def get_train_data(self):
        """Get all training embeddings and images.
        
        Don't use this function if not necessary.

        Returns:
            train_embeds: embeddings of training tags
            train_images: training images
        """           
        return self.train_embeds, self.images
   
    def get_test_data(self):
        """Get all testing embeddings.
        
        Returns:
            test_embeds: dictionary, where its key is test_id
                         and the corresponding value is embedding
        """
        
        return self.test_embeds

    def get_fixed_test_data(self, noise_dim=100, noise_type='normal'):
        """Get fixed noise and testing embeddings

        Args:
            noise_dim: dimension of noise
            noise_type: distribution of noise,
                        supports normal and uniform

        Returns:
            fixed_noise: dictionary, where its key is test_id
                         and the corresponding value is noise
            test_embeds: dictionary, where its key is test_id
                         and the corresponding value is embedding
        """
        if self.fixed_noise is None:
            if not noise_type in self.support_noise_type:
                print('unknown noise type \'{:s}\','
                      ' replaced with normal'.format(noise_type))
                noise_type = 'normal'

            self.fixed_noise = {}
            for id in self.test_embeds:
                self.fixed_noise[id] = self._get_noise(noise_type=noise_type,
                                                        shape=(noise_dim,))
        
        return self.fixed_noise, self.test_embeds
    
    def get_next_batch(self, batch_size=64, noise_dim=100,
                       noise_type='normal'):
        """Get batches for GAN.
        
        Args:
            batch_size: number of examples in a batch
            noise_dim: dimenson of noise
            noise_type: distribution for noise, 
                        supports 'normal' or 'uniform'
        
        Yields:
            real_imgs: shape (batch_size, 64, 64, 3)
                       real images
            noise: shape (batch_size, noise_dim)
            right_embeds: shape (batch_size, embed_dim)
                          embedding of right captions
            wrong_embeds: shape (batch_size, embed_dim)
                          embedding of wrong captions            
        """
        if not noise_type in self.support_noise_type:
            print('unknown noise type \'{:s}\','
                  ' replaced with normal'.format(noise_type))
            noise_type = 'normal'
        
        n_data = self.train_embeds.shape[0]
        perm_idx = np.arange(n_data)
        np.random.shuffle(perm_idx)
        
        n_batches = n_data // batch_size
        
        for batch_num in range(n_batches):
            start = batch_num * batch_size
            batch_indices = perm_idx[start:start+batch_size]
            
            real_imgs = self.images[batch_indices]
            real_imgs_augmented = next(self.data_gen.flow(real_imgs,
                                    batch_size=batch_size,
                                    shuffle=False))

            noise = self._get_noise(noise_type=noise_type, 
                                    shape=(batch_size, noise_dim))
            
            right_embeds = self.train_embeds[batch_indices]
            wrong_embeds = self._get_wrong_embeds(batch_indices)
            
            yield real_imgs_augmented, noise, right_embeds, wrong_embeds

