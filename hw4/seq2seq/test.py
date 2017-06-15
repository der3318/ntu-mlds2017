from data import data
from model import seq2seq
import argparse
import tensorflow as tf
import numpy as np
def main(args):

    Data = data(
            train_data_path=args.data_path,
            valid_data_path=args.valid_path,
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
            embedding_dim=args.embedding,
            use_ss=args.use_ss,
            # use_att=args.use_att,
            use_bn=args.use_bn,
            use_dropout=True,
            beam_size=args.beam_size,
            alpha_c=args.alpha_c,
            seed=3318
        )

    pred_words = model.build_predict_model(
        batch_size=1
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        if args.model_path != None:
            saver = tf.train.Saver()
            saver.restore(sess, args.model_path)
            print('Restore model done')

        def gen_sentence(x):
            x = x.lower().split()
            x = Data.get_indices_by_sentence(x)

            if len(x) + 2 >= args.nstep:
                x = x[:args.nstep - 2]

            x.insert(0, Data.get_index_by_word('<BOS>'))
            x.append(Data.get_index_by_word('<EOS>'))
            for _ in range(args.nstep - len(x)):
                x.append(Data.get_index_by_word('<PAD>'))
            # print(x)
            pred = model.predict(np.asarray([x]), sess, Data, predict_tensor=pred_words)
            # print(pred[0])
            y = []
            for i in pred[0]:
                y += [i]
                if i == 3:
                    break
            return ' '.join(Data.get_words_by_indices(y))

        if args.input_path == None:
            while True:
                x = input("Input: ")
                print(gen_sentence(x))
        else:
            output_file = open(args.output_path, 'w')
            with open(args.input_path, "r") as input_file:

                for line in input_file:
                    output_file.write("{}\n".format(gen_sentence(line)))
            output_file.close()





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
    parser.add_argument('--embedding',
                        help='embedding size',
                        default=130,
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

    parser.add_argument('--model_path',
                        help='resume model path if resuming model')


    parser.add_argument('--dict_path',
                        help='path of dictionary file',
                        default='./dict.txt')
    parser.add_argument('--data_path',
                        help='path of data file',
                        default='./data_train.npy')
    parser.add_argument('--valid_path',
                        help='path of data file',
                        default='./data_valid.npy')
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
    parser.add_argument('--input_path',
                        help='path of input file',
                        default=None)
    parser.add_argument('--output_path',
                        help='path of outptu file',
                        default='output.txt')

    args = parser.parse_args()

    main(args)
