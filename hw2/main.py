import argparse
import os
import numpy as np

import bleu

from preprocessing import data
from model import S2VTmodel

######################################
# Parameters specificly for this task
######################################

frames = 80
dim_image = 4096

######################################
# Paths
######################################

path = 'MLDS_hw2_data/'
model_path = './models/'

def main(args):
    Data = data(
            os.path.join(path, 'training_data/feat/'),
            os.path.join(path, 'training_label.json'),
            os.path.join(path, 'testing_data/feat/'),
            os.path.join(path, 'testing_public_label.json'),
            )
    
    Data.load()
    n_words = Data.get_vocab_size()

    model = S2VTmodel(
                n_hidden=args.hidden,
                n_step1=frames,
                n_step2=args.nstep,
                use_ss=args.use_ss,
                use_att=args.use_att,
                n_words=n_words,
                dim_image=dim_image,
                seed=3318
                )

    train_X, train_y, valid_X, valid_y = Data.gen_validation_data(cross=False, caption_num=10)
    
    if args.train:
        model.train(Data, train_X, train_y, valid_X, valid_y,
                    batch_size=args.batch_size,
                    learning_rate=args.rate,
                    epoch=args.epoch,
                    period=args.period,
                    name=args.name
                    )

    test_X, test_y = Data.gen_test_data()
    
    pred = model.predict(test_X, model_path=os.path.join(model_path, args.name))

    scores = []
    for p, ty in zip(pred, test_y):
        sent = Data.get_sentence_by_indices(p)
        score = bleu.eval(sent, ty)
        scores.append(score)

    print('\naverage bleu score overall:\n\t', np.mean(scores))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden', 
                        help='number of hidden units', 
                        default=512, 
                        type=int)

    parser.add_argument('--nstep', '-n', 
                        help='maximum length of captions', 
                        default=20, 
                        type=int)

    parser.add_argument('--batch_size', '-b', 
                        help='batch size', 
                        default=128, 
                        type=int)

    parser.add_argument('--rate', '-r', 
                        help='learning rate', 
                        default=1e-3, 
                        type=float)

    parser.add_argument('--epoch', '-e',
                        help='number of epoch to train', 
                        default=100, 
                        type=int)
    
    parser.add_argument('--period', '-p',
                        help='intervals between checkpoints', 
                        default=2, 
                        type=int)

    parser.add_argument('--use_ss',
                        help='whether to use schedule sampling',
                        action='store_true')
    
    parser.add_argument('--use_att',
                        help='whether to use attention',
                        action='store_true')
    
    parser.add_argument('--name',
                        help='model name to save', 
                        default='model')
    
    parser.add_argument('--train',
                        help='whether train the model or not',
                        action='store_true')

    args = parser.parse_args()

    main(args)
