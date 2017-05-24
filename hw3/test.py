from __future__ import print_function

import tensorflow as tf
import numpy as np
np.random.seed(3318)
import argparse

from os.path import join
import scipy.misc
import model_wgan
import data

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--z_dim', type=int, default=100,
                       help='Noise dimension')

    parser.add_argument('--y_dim', type=int, default=128,
    					help='Text feature dimension')

    parser.add_argument('--c_dim', type=int, default=3,
                       help='image channel')

    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')

    parser.add_argument('--img_size', type=int, default=64,
                       help='Image Size a, a x a')

    parser.add_argument('--gf_dim', type=int, default=64,
                       help='Number of conv in the first layer generator.')

    parser.add_argument('--df_dim', type=int, default=64,
                       help='Number of conv in the first layer discriminator.')

    parser.add_argument('--gfc_dim', type=int, default=128,
                       help='Dimension of gen untis for for fully connected layer 1024')

    parser.add_argument('--dfc_dim', type=int, default=128,
                       help='Dimension of dis untis for for fully connected layer 1024')

    parser.add_argument('--testing_text',type=str, default="./data/sample_testing_text.txt",
                       help='the path of testing text')

    parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

    args = parser.parse_args()
    return args

def test(args,total_data):
    gan = model_wgan.GAN(args,total_data.get_embed_dim())
    _, _, _, _, _ = gan.build_model("ass_gan")
    input_tensors, outputs = gan.build_sampler()

    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=config)
    
    saver = tf.train.Saver()
    if args.resume_model:
        saver.restore(sess, args.resume_model)

    test_caption = total_data.get_test_data()

    for sample_id in range(1,6):
        test_noise = { id:np.random.normal(0,1,100) for id in sorted(test_caption) }
        generated_images = {}
        for id in sorted(test_caption):
            gen = sess.run(outputs['generator'],
                            feed_dict={
                                input_tensors['t_real_caption']: [test_caption[id]],
                                input_tensors['t_z']: [test_noise[id]]
                            })
            generated_images[id] = gen[0]

        print ("Saving Images, Model")
        save_for_vis(generated_images,sample_id)

def save_for_vis(generated_images, sample_id):
    for id in generated_images:
        fake_images_255 = (generated_images[id])
        scipy.misc.imsave('samples/sample_{}_{}.jpg'.format(id, sample_id), fake_images_255)

def main():
    args = arg_parse()
    total_data = data.Data(train_file='data/tags_clean.csv', test_file=args.testing_text, train_embed_path='data/embed_onehot.npy', test_embed_path='data/test_onehot', test_only=True)
    test(args,total_data)

if __name__ == '__main__':
    main()