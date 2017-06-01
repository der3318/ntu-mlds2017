# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
import tensorflow as tf
from datetime import datetime
from model import seq2seq
from data import data
def main(args):

    Data = data(
            train_data_path=args.data_path,
            dict_path=args.dict_path,
            n_step=args.nstep
        )

    n_words = Data.get_vocab_size()
    print(n_words)


        # dropout預設皆使用， att, bn, beam search皆未實作
    model = seq2seq(
            n_layers=args.layers,
            n_hidden=args.hidden,
            n_step1=args.nstep,
            n_step2=args.nstep,
            dim_input=n_words,
            dim_output=n_words,
            use_ss=args.use_ss,
            use_att=args.use_att,
            use_bn=args.use_bn,
            use_dropout=True,
            beam_size=args.beam_size,
            alpha_c=args.alpha_c,
            seed=3318
        )


    # model.build_model(batch_size=args.batch_size, is_training=True)
    # train_X, train_y, valid_X, valid_y = Data.gen_train_data(test_ratio=0.01)
    model.train(
        # X=train_X,
        # y=train_y,
        # valid_X=valid_X,
        # valid_y=valid_y,
        Data=Data,
        batch_size=args.batch_size,
        learning_rate=args.rate,
        epoch=args.epoch,
        period=args.period,
        name=args.name,
        dropout_rate=0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden',
                        help='number of hidden units',
                        default=256,
                        type=int)
    parser.add_argument('--layers',
                        help='number of hidden layers',
                        default=2,
                        type=int)
    parser.add_argument('--dim_input',
                        help='dimensions of input vocabulary size',
                        default=5004,
                        type=int)
    parser.add_argument('--dim_output',
                        help='dimensions of output vocabulary size',
                        default=5004,
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
    parser.add_argument('--dict_path',
                        help='path of dictionary file',
                        default='./dict.txt')
    parser.add_argument('--data_path',
                        help='path of data file',
                        default='./open_subtitles.npy')
    parser.add_argument('--period', '-p',
                        help='intervals between checkpoints',
                        default=2,
                        type=int)
    #
    parser.add_argument('--use_ss',
                        help='whether to use schedule sampling',
                        action='store_true')

    parser.add_argument('--use_att',
                        help='whether to use attention',
                        action='store_true')

    parser.add_argument('--use_bn',
                        help='whether to use batch normalization',
                        action='store_true')

    parser.add_argument('--beam_size',
                        help='number of beams for beam search',
                        default=1,
                        type=int)

    parser.add_argument('--alpha_c',
                        help='regularization parameter for attention',
                        default=0.0,
                        type=float)

    parser.add_argument('--name',
                        help='model name to save',
                        default='model_{}'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))
    #
    # parser.add_argument('--model_epoch',
    #                     help='the specific checkpoint of the model',
    #                     default=None,
    #                     type=int)
    #
    # parser.add_argument('--train',
    #                     help='whether train the model or not',
    #                     action='store_true')

    args = parser.parse_args()

    main(args)
