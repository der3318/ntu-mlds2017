import numpy as np
import os
import sys
import argparse

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib
from utility import img2npy, read_tags, read_test_texts


def preprocess_img(args):
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


def preprocess_onehot(args):
    save_dir_path = './data/'
    if not os.path.exists(save_dir_path):
        print('creating data directory')
        os.makedirs(save_dir_path, exist_ok=True)
    
    keywords = np.load('./data/keywords.npy')
    tags = read_tags()
    tags = [list(filter(lambda x: x in keywords, tags[id])) for id in range(33431)]

    mlb = MultiLabelBinarizer()
    mlb.fit(tags)
    joblib.dump(mlb, 'binarizer.pkl')

    tags_onehot = mlb.transform(tags)
    np.save('data/embed_onehot.npy', tags_onehot)

    # testing texts
    texts = read_test_texts(args.test_file)
    save_dir_path = './data/test_onehot/'
    if not os.path.exists(save_dir_path):
        print('creating data directory')
        os.makedirs(save_dir_path)

    for id in texts:
        save_file_path = os.path.join(save_dir_path,
                                    'embedding_'+str(id)+'.npy')
        text = list(filter(lambda x: x in texts[id], keywords))
        print(text)
        vec = mlb.transform([text])
        np.save(save_file_path, vec[0])


def preprocess_test_embed(test_file):
    mlb = joblib.load('binarizer.pkl')
    keywords = np.load('./data/keywords.npy')
    # testing texts
    texts = read_test_texts(test_file)
    save_dir_path = './data/test_onehot/'
    if not os.path.exists(save_dir_path):
        print('creating data directory')
        os.makedirs(save_dir_path)

    for id in texts:
        save_file_path = os.path.join(save_dir_path,
                                    'embedding_'+str(id)+'.npy')
        text = list(filter(lambda x: x in texts[id], keywords))
        print(text)
        vec = mlb.transform([text])
        np.save(save_file_path, vec[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        help='directory contains face jpgs')
    parser.add_argument('test_file',
                        help='path to testing text file')

    args = parser.parse_args()
    
    #preprocess_img(args)
    #preprocess_onehot(args)
    preprocess_test_embed(args.test_file)
