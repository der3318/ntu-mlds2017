from __future__ import print_function

import os
import sys
import numpy as np

from scipy.spatial.distance import cosine
from utility import read_tags, read_test_texts


class Data:
    
    support_noise_type = ['normal', 'uniform']
    
    def __init__(self, test_file,
                 test_embed_path='./data/test',
                 test_only=False, seed=3318):
        """
        Args:
            test_file: path to testing text file
            test_embed_path: path to testing embeddings
            test_only: whether this data class if for
                       test only
            seed: seed for reproducing

        """
        self.test_file = test_file
        self.test_embed_path = test_embed_path
        self.test_only = test_only
        self.seed = seed
        self.fixed_noise = None
        
        np.random.seed(self.seed)
        
        self.test_texts = read_test_texts(test_file)
        self.train_tags = read_tags(min_count=1)
        
        if not self.test_only:
            self._get_images()
        
        self._get_embeds(self.test_only)

    def _get_images(self):
        print('loading images... ', end='')
        self.images = np.load('./data/images.npy', mmap_mode='r')
        print('done')
        
        return
        
    def _get_embeds(self, test_only=False):
        print('loading embeddings... ', end='')
 
        if not test_only:
            # load training embeddings
            self.train_embeds = np.load('./data/embed_skipthoughts.npy', 
                                        mmap_mode='r')

        # load testing embeddings
        self.test_embeds = {}
        for f in os.listdir(self.test_embed_path):
            file_path = os.path.join(self.test_embed_path, f)
            test_id = f.replace('.npy', '').replace('embedding_', '')
            self.test_embeds[test_id] = np.load(file_path)
        
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
        wrong_embed = self.train_embeds[np.random.randint(n_embeds)]
        
        while cosine(embed, wrong_embed) > 0.5:
            wrong_embed = self.train_embeds[np.random.randint(n_embeds)]
        
        return wrong_embed
        
    def _get_wrong_embeds(self, indices):
        return np.array([self.__get_wrong_embeds(i) for i in indices])
        
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
            fixed_noise: shape (1, noise_dim)
                         a fixed noise, used for all testing data
            test_embeds: dictionary, where its key is test_id
                         and the corresponding value is embedding
        """
        if self.fixed_noise is None:
            if not noise_type in self.support_noise_type:
                print('unknown noise type \'{:s}\','
                      ' replaced with normal'.format(noise_type))
                noise_type = 'normal'

            self.fixed_noise = self._get_noise(noise_type=noise_type,
                                               shape=(1, noise_dim))
        
        return self.fixed_noise, self.test_embeds
    
    def get_next_batch_generator(self, batch_size=64, noise_dim=100,
                                 noise_type='normal'):
        """Get batches for generator.
        
        Args:
            batch_size: number of examples in a batch
            noise_dim: dimenson of noise
            noise_type: distribution for noise, 
                        supports 'normal' or 'uniform'
        
        Yields:
            noise: shape (batch_size, noise_dim)
            embeds: shape (batch_size, embed_dim)
            batch_indices: shape (batch_size, )
                           the indices of this batch
            
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
            
            noise = self._get_noise(noise_type=noise_type, 
                                    shape=(batch_size, noise_dim))
            embeds = self.train_embeds[batch_indices]
            
            yield noise, embeds, batch_indices
    
    def get_next_batch_discriminator(self, gen_imgs, indices,
                                     batch_size=64,
                                     real_value=1, fake_value=0):
        """Get batches for discriminator
        
        Args:
            gen_imgs: shape (n_samples, 64, 64, 3)
                      images generated by generator
            indices: shape (n_samples, )
                     indices of samples
            batch_size: number of example in a batch
            real_value: target value for (real image, right text)
            fake_value: target value for (real image, wrong text)
                        or (fake image, right text)
        
        Yields:
            imgs: shape (batch_size, 64, 64, 3)
                  images for discriminator input
            embeds: shape (batch_size, embed_dim)
                    embeddings for discriminator input
            y: shape (batch_size, 1)
               target value for discriminator output
        """
        gen_imgs = np.array(gen_imgs)
        indices = np.array(indices)
        
        n_samples = gen_imgs.shape[0]
        perm_idx = np.arange(n_samples*3)
        np.random.shuffle(perm_idx)
        
        n_batches = (n_samples*3) // batch_size
        
        for batch_num in range(n_batches):
            start = batch_num * batch_size
            batch_indices = perm_idx[start:start+batch_size]
            
            types = batch_indices // n_samples
            idx = batch_indices % n_samples
            
            imgs = np.concatenate(
                    (self.images[indices[idx[types==0]]],
                     self.images[indices[idx[types==1]]],
                     gen_imgs[idx[types==2]]), axis=0)
            embeds = np.concatenate(
                        (self.train_embeds[indices[idx[types==0]]],
                         self._get_wrong_embeds(indices[idx[types==1]]),
                         self.train_embeds[indices[idx[types==2]]]), axis=0)
            y = np.concatenate((np.full((types==0).sum(), real_value),
                                np.full((types==1).sum(), fake_value),
                                np.full((types==2).sum(), fake_value)), axis=0)
            y = y.reshape((-1, 1))
            
            yield imgs, embeds, y
