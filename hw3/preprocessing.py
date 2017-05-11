import numpy as np
import os
import sys
import argparse

from utility import img2npy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='directory contains face jpgs')

    args = parser.parse_args()
    
    save_dir_path = './data/'
    if not os.path.exists(save_dir_path):
        print('creating data directory')
        os.makedirs(save_dir_path, exist_ok=True)

    dir_path = args.data_dir
    images = []
    for i in range(33431):
        path = os.path.join(dir_path, str(i)+'.jpg')

        img = img2npy(path)
        images.append(img)

        if i % 100 == 0:
            print('processed {:d} images'.format(i), end='\r')

    print('saving all images into images.npy')
    np.save(os.path.join(save_dir_path, 'images.npy'), images)
