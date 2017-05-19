import tensorflow as tf
import numpy as np
from ops import *

class GAN():
    def __init__(self,args,text_dim):
        self.batch_size = args.batch_size
        self.img_size = args.img_size

        self.data_y_dim = text_dim
        self.y_dim = args.y_dim
        self.z_dim = args.z_dim
        self.c_dim = args.c_dim

        self.gfc_dim = args.gfc_dim
        self.gf_dim = args.gf_dim

        self.df_dim = args.df_dim
        self.dfc_dim = args.dfc_dim

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

    def build_model(self):
        t_real_image = tf.placeholder('float32', [self.batch_size,self.img_size, self.img_size, 3 ], name = 'real_image')
        t_false_caption = tf.placeholder('float32', [self.batch_size, self.data_y_dim], name = 'false_caption_input')
        t_real_caption = tf.placeholder('float32', [self.batch_size, self.data_y_dim], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [self.batch_size, self.z_dim])

        fake_image = self.generator(t_z, t_real_caption)
        
        disc_real_image, disc_real_image_logits   = self.discriminator(t_real_image, t_real_caption)
        disc_real_image, disc_false_caption_logits   = self.discriminator(t_real_image, t_false_caption, reuse = True)
        disc_fake_image, disc_fake_image_logits   = self.discriminator(fake_image, t_real_caption, reuse = True)
        
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.ones_like(disc_fake_image)))
        
        d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_image_logits, labels=tf.ones_like(disc_real_image)))
        d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_false_caption_logits, labels=tf.zeros_like(disc_real_image)))
        d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.zeros_like(disc_fake_image)))

        d_loss = d_loss1 + d_loss2 + d_loss3

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        input_tensors = {
            't_real_image' : t_real_image,
            't_false_caption' : t_false_caption,
            't_real_caption' : t_real_caption,
            't_z' : t_z
        }

        variables = {
            'd_vars' : d_vars,
            'g_vars' : g_vars
        }

        loss = {
            'g_loss' : g_loss,
            'd_loss' : d_loss
        }

        outputs = {
            'generator' : fake_image
        }

        checks = {
            'd_loss1': d_loss1,
            'd_loss2': d_loss2,
            'd_loss3' : d_loss3,
            'disc_real_image_logits' : disc_real_image_logits,
            'disc_false_caption_logits' : disc_false_caption_logits,
            'disc_fake_image_logits' : disc_fake_image_logits
        }
        
        return input_tensors, variables, loss, outputs, checks

    def build_sampler(self):
    	img_size = self.options['image_size']
		t_real_caption = tf.placeholder('float32', [self.batch_size, self.data_y_dim], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.batch_size, self.z_dim])
		fake_image = self.sampler(t_z, t_real_caption)
		
		input_tensors = {
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}
		
		outputs = {
			'generator' : fake_image
		}

		return input_tensors, outputs

    def generator(self, z, y):
        with tf.variable_scope("generator") as scope:
            s= int(self.img_size)
            s2, s4 = int(s/2),int(s/4)

            y = lrelu( linear(y, self.y_dim, 'g_embedding') )

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,[self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def sampler(self, z, y):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s= int(self.img_size)
            s2, s4 = int(s/2),int(s/4)

            y = lrelu( linear(y, self.y_dim, 'g_embedding') )

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
            h0 = concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def discriminator(self, image, y , reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            y = lrelu( linear(y, self.y_dim, 'd_embedding') )

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])      
            h1 = concat([h1, y], 1)
            
            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = concat([h2, y], 1)

            h3 = linear(h2, 1, 'd_h3_lin')
            
            return tf.nn.sigmoid(h3), h3

