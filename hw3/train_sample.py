from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse

from os.path import join
import scipy.misc
import model
import data
import random
import json
import os
import shutil

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

    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Momentum for Adam Update')

    parser.add_argument('--epochs', type=int, default=600,
                       help='Max number of epochs')

    parser.add_argument('--save_every', type=int, default=30,
                       help='Save Model/Samples every x iterations over batches')

    parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--img_dir',type=str, default=None,
                       help='Dir for visualization in every 30 epoch')

    args = parser.parse_args()
    return args

def train(args,total_data):
    gan = model.GAN(args,total_data.get_embed_dim())
    input_tensors, variables, loss, outputs, checks = gan.build_model()

    d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
    g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])
    
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    tf.global_variables_initializer().run()
    tf.set_random_seed(33)
    
    #saver = tf.train.Saver()
    if args.resume_model:
        saver.restore(sess, args.resume_model)
    progress1 = ["[| \x1b[0;32;40m>\x1b[0m", "[| \x1b[0;32;40m>>\x1b[0m", "[| \x1b[0;32;40m>>>\x1b[0m", "[| \x1b[0;32;40m>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>>\x1b[0m"]
    progress2 = ["\x1b[0;31;40m---------\x1b[0m |]", "\x1b[0;31;40m--------\x1b[0m |]", "\x1b[0;31;40m-------\x1b[0m |]", "\x1b[0;31;40m------\x1b[0m |]", "\x1b[0;31;40m-----\x1b[0m |]", "\x1b[0;31;40m----\x1b[0m |]", "\x1b[0;31;40m---\x1b[0m |]", "\x1b[0;31;40m--\x1b[0m |]", "\x1b[0;31;40m-\x1b[0m |]", " |]"]
    
    test_noise, test_caption = total_data.get_fixed_test_data()

    for i in range(args.epochs):
        for batch_no,(real_images, z_noise, real_caption, false_caption) in enumerate(total_data.get_next_batch()):
            
            # DISCR UPDATE
            discriminator_update_time = 1
            for j in range(discriminator_update_time):
                check_ts = [ checks['d_loss1'], checks['d_loss2'], checks['d_loss3']]
                _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
                    feed_dict = {
                        input_tensors['t_real_image'] : real_images,
                        input_tensors['t_false_caption'] : false_caption,
                        input_tensors['t_real_caption'] : real_caption,
                        input_tensors['t_z'] : z_noise,
                    })

            # GEN UPDATE TWICE, to make sure d_loss does not go to 0
            generator_update_time = 2
            # GEN UPDATE
            for j in range(generator_update_time):
                _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
                    feed_dict = {
                        input_tensors['t_real_image'] : real_images,
                        input_tensors['t_false_caption'] : false_caption,
                        input_tensors['t_real_caption'] : real_caption,
                        input_tensors['t_z'] : z_noise,
                    })

            
            pro = (batch_no*args.batch_size)*10 // total_data.get_data_length()
            print("\r0%" + progress1[pro] + progress2[pro] + "100% - Epoch: " + str(i) + "/" + str(args.epochs) + " - Loss(g_loss/d_loss): " + str(g_loss) + "/" + str(d_loss)+" ", end = "\r", flush=True)
       
        print()
        generated_images = {}
        for id in sorted(test_caption):
            gen = sess.run(outputs['generator'],
                            feed_dict={
                                input_tensors['t_real_caption']: [test_caption[id]]*args.batch_size,
                                input_tensors['t_z']: [test_noise[id]]*args.batch_size
                            })
            generated_images[id] = gen[0]

        save_for_vis(args.img_dir, generated_images, i)
        #if i%5 == 0:
            #save_path = saver.save(sess, "./data/model/model_after_epoch_{}.ckpt".format(i))

def save_for_vis(data_dir, generated_images, global_step):
    #shutil.rmtree( join(data_dir, 'samples') )
    #os.makedirs( join(data_dir, 'samples') )
    for id in generated_images:
        fake_images_255 = (generated_images[id])
        scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}_{}.jpg'.format(id, global_step)), fake_images_255)

def main():
    args = arg_parse()
    total_data = data.Data(train_file='data/tags_clean.csv', test_file='data/sample_testing_text.txt', train_embed_path='data/embed_onehot.npy', test_embed_path='data/test_onehot')
    train(args,total_data)

if __name__ == '__main__':
    main()
