from __future__ import print_function

import sys
import os
import argparse
import numpy as np

from embedding import Embedding
from utility import read_tags, read_test_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_file',
                        help='path to testing text file')
    parser.add_argument('--train',
                        help='preprocess for training or only for testing')

    args = parser.parse_args()
    
    # preprocess skip-thoughts embeddings
    embed = Embedding('skipthoughts')

    if args.train:
        # training tags
        tags = read_tags(min_count=1)
        tags_sent = [' '.join(tags[id]) if len(tags[id])>0 else '.' for id in range(33431)]
        vecs = embed.encode(tags_sent)

        save_dir_path = './data/'
        np.save(os.path.join(save_dir_path, 'embed_skipthoughts.npy'), vecs)


    # testing texts
    texts = read_test_texts(args.test_file)
    save_dir_path = './data/test/'
    if not os.path.exists(save_dir_path):
        print('creating data directory')
        os.makedirs(save_dir_path)

    for id in texts:
        save_file_path = os.path.join(save_dir_path,
                                    'embedding_'+str(id)+'.npy')
        text = texts[id] if len(texts[id])>0 else '.'
        vec = embed.encode([text])
        np.save(save_file_path, vec)



