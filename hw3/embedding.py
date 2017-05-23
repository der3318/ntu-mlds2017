import sys
import os
import numpy as np


class Embedding:
    def __init__(self,
                 train_embed_path='./data/embed_skipthoughts.npy',
                 test_embed_path='./data/test/'):
        """
        Arguments:
        train_embed_path    path for the training embedding
        test_embed_path     path for the directory contains
                            testing embedding

        """
        self.train_embed_path = train_embed_path
        self.test_embed_path = test_embed_path

        return

    def get_embeds(self):
        # load training embeddings
        train_embeds = np.load(self.train_embed_path, mmap_mode='r')

        # load testing embeddings
        test_embeds = {}
        for f in os.listdir(self.test_embed_path):
            file_path = os.path.join(self.test_embed_path, f)
            test_id = f.replace('.npy', '').replace('embedding_', '')
            test_embeds[test_id] = np.load(file_path)

        return train_embeds, test_embeds

