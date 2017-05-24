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
        self.g_bn3 = batch_norm(name='g_bn3')

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

    def build_model(self,model_type):
        t_real_image = tf.placeholder('float32', [self.batch_size,self.img_size, self.img_size, 3 ], name = 'real_image')
        t_false_caption = tf.placeholder('float32', [self.batch_size, self.data_y_dim], name = 'false_caption_input')
        t_real_caption = tf.placeholder('float32', [self.batch_size, self.data_y_dim], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [self.batch_size, self.z_dim])

        fake_image = self.generator(t_z, t_real_caption)
        
        disc_real_image, disc_real_image_logits   = self.discriminator(t_real_image, t_real_caption)
        disc_real_image, disc_false_caption_logits   = self.discriminator(t_real_image, t_false_caption, reuse = True)
        disc_fake_image, disc_fake_image_logits   = self.discriminator(fake_image, t_real_caption, reuse = True)

        if model_type == "dcgan":
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.ones_like(disc_fake_image)))
        
            d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_image_logits, labels=tf.ones_like(disc_real_image)))
            d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_false_caption_logits, labels=tf.zeros_like(disc_real_image)))
            d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.zeros_like(disc_fake_image)))

            d_loss = d_loss1 + d_loss2 + d_loss3
        else:
            g_loss = tf.reduce_mean(disc_fake_image_logits)
        
            d_loss1 = tf.reduce_mean(disc_real_image_logits)
            d_loss2 = tf.reduce_mean(disc_false_caption_logits)
            d_loss3 = tf.reduce_mean(disc_fake_image_logits)

            d_loss = d_loss1 - (d_loss2 + d_loss3)/2

            if model_type == "wgan_v2":
                epsilon = tf.random_uniform([], 0.0, 1.0)
                x_hat = epsilon * t_real_image + (1 - epsilon) * fake_image
                _,d_hat = self.discriminator(x_hat, t_real_caption, reuse = True)

                ddx = tf.gradients(d_hat, x_hat)[0]
                ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
                ddx = tf.reduce_mean(tf.square(ddx - 1.0) * 10)

                d_loss = d_loss + ddx

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
        t_real_caption = tf.placeholder('float32', [1, self.data_y_dim], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [1, self.z_dim])
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
            s2, s4, s8, s16 = int(s/2),int(s/4),int(s/8),int(s/16)

            y = lrelu( linear(y, self.y_dim, 'g_embedding') )

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            z = concat([z, y], 1)

            z = linear(z,self.gf_dim*8*s16*s16,'g_h0_lin')

            h0 = tf.reshape(z, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0))
            
            h1 = deconv2d(h0, [ self.batch_size, s8, s8,  self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))
            
            h2 = deconv2d(h1, [ self.batch_size, s4, s4,  self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))
            
            h3 = deconv2d(h2, [ self.batch_size, s2, s2,  self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))
            
            h4 = deconv2d(h3, [ self.batch_size, s, s, 3], name='g_h4')
        
        return (tf.tanh(h4))

    def sampler(self, z, y):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s= int(self.img_size)
            s2, s4, s8, s16 = int(s/2),int(s/4),int(s/8),int(s/16)

            y = lrelu( linear(y, self.y_dim, 'g_embedding') )

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            z = concat([z, y], 1)

            z = linear(z,self.gf_dim*8*s16*s16,'g_h0_lin')

            h0 = tf.reshape(z, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0))
            
            h1 = deconv2d(h0, [ 1, s8, s8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))
            
            h2 = deconv2d(h1, [ 1, s4, s4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))
            
            h3 = deconv2d(h2, [ 1, s2, s2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))
            
            h4 = deconv2d(h3, [ 1, s, s, 3], name='g_h4')
        
        return (tf.tanh(h4))

    def discriminator(self, image, y , reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu( conv2d(image, self.df_dim, name = 'd_h0_conv')) #32
            h1 = lrelu( self.d_bn1(conv2d(h0, self.df_dim*2, name = 'd_h1_conv'))) #16
            h2 = lrelu( self.d_bn2(conv2d(h1, self.df_dim*4, name = 'd_h2_conv'))) #8
            h3 = lrelu( self.d_bn3(conv2d(h2, self.df_dim*8, name = 'd_h3_conv'))) #4

            y = lrelu( linear(y, self.y_dim, 'd_embedding'))
            y = tf.expand_dims(y,1)
            y = tf.expand_dims(y,2)
            y = tf.tile(y, [1,4,4,1], name='tiled_embeddings')
            
            h3_concat = tf.concat( [h3, y], 3, name='h3_concat')

            h3_new = lrelu(self.d_bn4(conv2d(h3_concat, self.df_dim*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4
        
            h4 = linear(tf.reshape(h3_new, [self.batch_size, -1]), 1, 'd_h3_lin')
        
        return tf.nn.sigmoid(h4), h4

