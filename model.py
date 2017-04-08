from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class DualNet(object):
    def __init__(self, sess, image_size=128, batch_size=1,fcn_filter_dim = 64,  \
                 input_channels_A = 3, input_channels_B = 3, dataset_name='facades', \
                 checkpoint_dir=None, lambda_A = 200, lambda_B = 200, lambda_pair=200,\
                 sample_dir=None, loss_metric = 'L1', network_type='fcn_1', use_labeled_data=False,\
                 clamp = 0.01, n_critic = 3, flip = False):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training. [1]
            image_size: (optional) The resolution in pixels of the images. [128]
            fcn_filter_dim: (optional) Dimension of fcn filters in first conv layer. [64]
            input_channels_A: (optional) Dimension of input image color of Network A. For grayscale input, set to 1. [3]
            input_channels_B: (optional) Dimension of output image color of Network B. For grayscale input, set to 1. [3]
        """
        self.clamp = clamp
        self.n_critic = n_critic
        self.df_dim = fcn_filter_dim
        self.flip = flip
        self.use_labeled_data = (use_labeled_data == 'semi')
        
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_pair = lambda_pair
        
        self.sess = sess
        self.is_grayscale_A = (input_channels_A == 1)
        self.is_grayscale_B = (input_channels_B == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        #self.L1_lambda = L1_lambda

        self.fcn_filter_dim = fcn_filter_dim
        self.network_type = network_type
        
        self.input_channels_A = input_channels_A
        self.input_channels_B = input_channels_B
        self.loss_metric = loss_metric

        # batch normalization : deals with poor initialization helps gradient flow

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        
        #directory name for output and logs saving
        self.dir_name = \
        "%s-batch_sz_%s-img_sz_%s-fltr_dim_%d-%s-%s-lambda_ABp_%s_%s_%s-c_%s-n_critic_%s-semi_%s" % \
        (self.dataset_name, self.batch_size, self.image_size,self.fcn_filter_dim,\
        self.loss_metric, self.network_type, self.lambda_A, self.lambda_B, \
        self.lambda_pair, self.clamp, self.n_critic, self.use_labeled_data) 
        
        self.build_model()

    def build_model(self):
    ###    define place holders
        self.real_A = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_channels_A ],
                                        name='input_images_of_A_network')
        self.real_B = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_channels_B ],
                                        name='input_images_of_B_network')
        self.real_PA = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_channels_A ],
                                        name='input_images_of_A_network')
        self.real_PB = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_channels_B ],
                                        name='input_images_of_B_network')
    ###  define graphs
        #with tf.device('/gpu:0'):
        self.translated_A = self.A_g_net(self.real_A, reuse = False)
        self.A_D_predictions = self.A_d_net(self.translated_A, reuse = False)
    
        #with tf.device('/gpu:1'):
        self.translated_B = self.B_g_net(self.real_B, reuse = False)
        self.B_D_predictions = self.B_d_net(self.translated_B, reuse = False)
        #self.predictions_PAB_pair = self.C_d_net(tf.concat(3, [self.real_PA,self.real_PB]), reuse = False)
    
    ### define loss
        self.recover_A = self.B_g_net(self.translated_A, reuse = True)
        self.recover_B = self.A_g_net(self.translated_B, reuse = True)
        #self.translated_PA = self.A_g_net(self.real_PA, reuse = True)
        #self.translated_PB = self.B_g_net(self.real_PB, reuse = True)
        
        if self.loss_metric == 'L1':
            self.A_loss = tf.reduce_mean(tf.abs(self.recover_A - self.real_A))
            self.B_loss = tf.reduce_mean(tf.abs(self.recover_B - self.real_B))
            #self.loss_PAB  = tf.reduce_mean(tf.abs(self.real_PA - self.translated_PB)) + \
            #tf.reduce_mean(tf.abs(self.real_PB - self.translated_PA))
        elif self.loss_metric == 'L2':
            self.A_loss = tf.reduce_mean(tf.square(self.recover_A - self.real_A))
            self.B_loss = tf.reduce_mean(tf.square(self.recover_B - self.real_B))
            #self.loss_PAB = tf.reduce_mean(tf.square(self.real_PA - self.translated_PB)) +\
            #tf.reduce_mean(tf.square(self.real_PB - self.translated_PA))
        
        self.A_D_predictions_ = self.A_d_net(self.real_B, reuse = True)
        self.A_d_loss_real = tf.reduce_mean(-self.A_D_predictions_)
        self.A_d_loss_fake = tf.reduce_mean(self.A_D_predictions) # + tf.reduce_mean(self.A_D_predictions_2)
        
        self.A_d_loss = self.A_d_loss_fake + self.A_d_loss_real
        self.A_g_loss = tf.reduce_mean(-self.A_D_predictions) + self.lambda_B * (self.B_loss )

        self.B_D_predictions_ = self.B_d_net(self.real_A, reuse = True)
        self.B_d_loss_real = tf.reduce_mean(-self.B_D_predictions_)
        self.B_d_loss_fake = tf.reduce_mean(self.B_D_predictions) #+ tf.reduce_mean(self.B_D_predictions_2)
        
        self.B_d_loss = self.B_d_loss_fake + self.B_d_loss_real
        self.B_g_loss = tf.reduce_mean(-self.B_D_predictions) + self.lambda_A * (self.A_loss )

        
        #predictions_A_pair = self.C_d_net(tf.concat(3, [self.real_A,self.translated_A]), reuse = True)
        #predictions_B_pair = self.C_d_net(tf.concat(3, [self.translated_B,self.real_B]), reuse = True)
        #predictions_PA_pair = self.C_d_net(tf.concat(3, [self.real_PA,self.translated_PA]), reuse = True)
        #predictions_PB_pair = self.C_d_net(tf.concat(3, [self.translated_PB,self.real_PB]), reuse = True)
        
        
        #self.C_d_loss_fake = tf.reduce_mean(predictions_A_pair)+ \
        #    tf.reduce_mean(predictions_B_pair)+ \
        #    tf.reduce_mean(predictions_PA_pair)+ \
        #    tf.reduce_mean(predictions_PB_pair)
        #self.C_d_loss_real = tf.mul(4.0,  tf.reduce_mean(-self.predictions_PAB_pair)) 
        #self.C_d_loss = self.C_d_loss_fake + self.C_d_loss_real\
         

        #self.g_loss_pair = \
        #tf.reduce_mean(-predictions_A_pair)+ \
        #tf.reduce_mean(-predictions_B_pair)+ \
        #tf.reduce_mean(-predictions_PA_pair)+ \
        #tf.reduce_mean(-predictions_PB_pair)+ \
        #self.lambda_pair * self.loss_PAB 
        
        #if self.use_labeled_data:
            #self.d_loss = self.A_d_loss + self.B_d_loss + self.C_d_loss
            #self.g_loss = self.A_g_loss + self.B_g_loss + self.g_loss_pair
        #else:
        self.d_loss = self.A_d_loss + self.B_d_loss
        self.g_loss = self.A_g_loss + self.B_g_loss
        """
        self.translated_A_sum = tf.summary.image("translated_A", self.translated_A)
        self.translated_B_sum = tf.summary.image("translated_B", self.translated_B)
        self.recover_A_sum = tf.summary.image("recover_A", self.recover_A)
        self.recover_B_sum = tf.summary.image("recover_B", self.recover_B)
        """
        ### define summary
        self.A_d_loss_sum = tf.summary.scalar("A_d_loss", self.A_d_loss)
        self.A_loss_sum = tf.summary.scalar("A_loss", self.A_loss)
        self.B_d_loss_sum = tf.summary.scalar("B_d_loss", self.B_d_loss)
        self.B_loss_sum = tf.summary.scalar("B_loss", self.B_loss)

        
        self.A_g_loss_sum = tf.summary.scalar("A_g_loss", self.A_g_loss)
        self.B_g_loss_sum = tf.summary.scalar("B_g_loss", self.B_g_loss)

        self.d_loss_sum = tf.summary.merge([self.A_d_loss_sum, self.B_d_loss_sum])
        self.g_loss_sum = tf.summary.merge([self.A_g_loss_sum, self.B_g_loss_sum, self.A_loss_sum, self.B_loss_sum])
        
        
        ## define trainable variables
        t_vars = tf.trainable_variables()

        self.A_d_vars = [var for var in t_vars if 'A_d_' in var.name]
        self.B_d_vars = [var for var in t_vars if 'B_d_' in var.name]
        #self.C_d_vars = [var for var in t_vars if 'C_d_' in var.name]
        
        self.A_g_vars = [var for var in t_vars if 'A_g_' in var.name]
        self.B_g_vars = [var for var in t_vars if 'B_g_' in var.name]
        
        if self.use_labeled_data:
            self.d_vars = self.A_d_vars + self.B_d_vars + self.C_d_vars
        else:
            self.d_vars = self.A_d_vars + self.B_d_vars
        self.g_vars = self.A_g_vars + self.B_g_vars
        
        self.saver = tf.train.Saver()


    def clip_trainable_vars(self, var_list):
        for var in var_list:
            self.sess.run(var.assign(tf.clip_by_value(var, -self.c, self.c)))

    def load_random_samples(self):
        #np.random.choice(
        data =np.random.choice(glob('./datasets/{}/val/A/*.jpg'.format(self.dataset_name)),self.batch_size)
        sample_A = [load_data(sample_file, image_size =self.image_size, flip = False) for sample_file in data]
        
        data = np.random.choice(glob('./datasets/{}/val/B/*.jpg'.format(self.dataset_name)),self.batch_size)
        sample_B = [load_data(sample_file, image_size =self.image_size, flip = False) for sample_file in data]

        sample_A_images = np.reshape(np.array(sample_A).astype(np.float32),(self.batch_size,self.image_size, self.image_size,-1))
        sample_B_images = np.reshape(np.array(sample_B).astype(np.float32),(self.batch_size,self.image_size, self.image_size,-1))
        return sample_A_images, sample_B_images

    def sample_shotcut(self, sample_dir, epoch, idx, batch_idxs):
        sample_A_imgs,sample_B_imgs = self.load_random_samples()
        
        Ag, recover_A_value, translated_A_value = self.sess.run([self.A_loss, self.recover_A, self.translated_A], feed_dict={self.real_A: sample_A_imgs, self.real_B: sample_B_imgs})
        
        Bg, recover_B_value, translated_B_value = self.sess.run([self.B_loss, self.recover_B, self.translated_B], feed_dict={self.real_A: sample_A_imgs, self.real_B: sample_B_imgs})

        save_images(translated_A_value, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_train_translated_A_{:02d}.png'.format(sample_dir,self.dir_name , epoch, idx, batch_idxs))
        save_images(recover_A_value, [self.batch_size,1],    './{}/{}/{:06d}_{:04d}_train_recover_A_{:02d}_.png'.format(sample_dir,self.dir_name, epoch,  idx, batch_idxs))
        
        save_images(translated_B_value, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_train_translated_B_{:02d}.png'.format(sample_dir,self.dir_name, epoch, idx,batch_idxs))
        save_images(recover_B_value, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_train_recover_B_epoch={:02d}.png'.format(sample_dir,self.dir_name, epoch, idx, batch_idxs))
        
        print("[Sample] A_loss: {:.8f}, B_loss: {:.8f}".format(Ag, Bg))

    def train(self, args):
        """Train Dual GAN"""
        decay = 0.9
        self.d_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay) \
                          .minimize(self.d_loss, var_list=self.d_vars)
                          
        self.g_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay) \
                          .minimize(self.g_loss, var_list=self.g_vars)          
        self.clip_d_vars_ops = [val.assign(tf.clip_by_value(val, -self.clamp, self.clamp)) for val in self.d_vars]
        tf.global_variables_initializer().run()

        self.writer = tf.summary.FileWriter("./logs/"+self.dir_name, self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            data_A = glob('./datasets/{}/train/A/*.jpg'.format(self.dataset_name))
            data_B = glob('./datasets/{}/train/B/*.jpg'.format(self.dataset_name))
            np.random.shuffle(data_A)
            np.random.shuffle(data_B)
            batch_idxs = min(len(data_A), len(data_B)) // (self.batch_size*self.n_critic)
            if self.use_labeled_data:
                data_PAB = glob('./datasets/{}/train/AB/*.jpg'.format(self.dataset_name))
                np.random.shuffle(data_PAB)
                batch_num_PAB = len(data_PAB)// (self.batch_size*self.n_critic)
            
            for idx in xrange(0, batch_idxs):
                imgA_batch_list = [self.load_training_imgs(data_A, idx+i) for i in xrange(self.n_critic)]
                imgB_batch_list = [self.load_training_imgs(data_B, idx+i) for i in xrange(self.n_critic)]
                if self.use_labeled_data:
                    imgPAB_batch_list = [self.load_pair_imgs(data_PAB, idx+i, batch_num_PAB) for i in xrange(self.n_critic)]
                else:
                    imgPAB_batch_list = []
                
                print("Epoch: [%2d] [%4d/%4d]"%(epoch, idx, batch_idxs))
                counter = counter + 1
                self.run_optim(imgA_batch_list, imgB_batch_list, imgPAB_batch_list, counter, start_time)

                if np.mod(counter, 100) == 1:
                    self.sample_shotcut(args.sample_dir, epoch, idx, batch_idxs)

                if np.mod(counter, args.save_latest_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def load_training_imgs(self, data, idx):
        batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = [load_data(batch_file, image_size =self.image_size, flip = self.flip) for batch_file in batch_files]
                
        batch_images = np.reshape(np.array(batch).astype(np.float32),(self.batch_size,self.image_size, self.image_size,-1))
        
        return batch_images
        
    def load_pair_imgs(self, data, idx, total_num):
        idx = idx % total_num 
        batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = [load_data_pair(batch_file, img_size =self.image_size) for batch_file in batch_files]
                
        batch_imgs_AB = np.reshape(np.array(batch).astype(np.float32),(self.batch_size,self.image_size, self.image_size,-1))
        
        return batch_imgs_AB
        
    def run_optim(self,imgA_batch_list, imgB_batch_list,imgPAB_batch_list,  counter, start_time):
        for i in xrange(self.n_critic):
            batch_A_images = imgA_batch_list[i]
            batch_B_images = imgB_batch_list[i]
            if self.use_labeled_data:
                batch_PAB_images = imgPAB_batch_list[i]
                _, Adfake,Adreal,Bdfake,Bdreal, Cdfake, Cdreal, Ad, Bd, Cd, summary_str = \
                        self.sess.run([self.d_optim, self.A_d_loss_fake, self.A_d_loss_real, self.B_d_loss_fake, self.B_d_loss_real, self.C_d_loss_fake, self.C_d_loss_real, self.A_d_loss, self.B_d_loss, self.C_d_loss, self.d_loss_sum], \
                        feed_dict = {self.real_A: batch_A_images, self.real_B: batch_B_images, self.real_PA:batch_PAB_images[:,:,:,0:self.input_channels_A], self.real_PB:batch_PAB_images[:,:,:,self.input_channels_A:]})
            else:
                _, Adfake,Adreal,Bdfake,Bdreal, Ad, Bd, summary_str = \
                        self.sess.run([self.d_optim, self.A_d_loss_fake, self.A_d_loss_real, self.B_d_loss_fake, self.B_d_loss_real, self.A_d_loss, self.B_d_loss, self.d_loss_sum], \
                        feed_dict = {self.real_A: batch_A_images, self.real_B: batch_B_images})
            #self.writer.add_summary(summary_str, counter)
            self.sess.run(self.clip_d_vars_ops)
        
        batch_A_images = imgA_batch_list[np.random.randint(self.n_critic, size=1)[0]]
        batch_B_images = imgB_batch_list[np.random.randint(self.n_critic, size=1)[0]]
        if self.use_labeled_data:
            batch_PAB_images = imgPAB_batch_list[np.random.randint(self.n_critic, size=1)[0]]
            _, Ag, Bg, gloss_pair, Aloss, Bloss, PAB_loss,\
            summary_str = self.sess.run([self.g_optim, self.A_g_loss, self.B_g_loss, \
            self.g_loss_pair, self.A_loss, self.B_loss, self.loss_PAB, self.g_loss_sum],\
            feed_dict={ self.real_A: batch_A_images, self.real_B: batch_B_images, \
            self.real_PA:batch_PAB_images[:,:,:,0:self.input_channels_A],\
            self.real_PB:batch_PAB_images[:,:,:,self.input_channels_A:]})
        else:
            _, Ag, Bg, Aloss, Bloss, summary_str = \
            self.sess.run([self.g_optim, self.A_g_loss, self.B_g_loss, self.A_loss, \
            self.B_loss, self.g_loss_sum], feed_dict={ self.real_A: batch_A_images, \
            self.real_B: batch_B_images})
            Cdfake = Cdreal = Cd = gloss_pair = PAB_loss = 0.0

        #self.writer.add_summary(summary_str, counter)
        print("time: %4.4f, A_d_loss: %.2f, A_g_loss: %.2f, B_d_loss: %.2f, B_g_loss: %.2f, C_d_loss: %.2f, g_loss_pair: %.2f, A_loss: %.5f, B_loss: %.5f, PAB_loss: %.5f" \
                    % (time.time() - start_time, Ad,Ag,Bd,Bg, Cd, gloss_pair, \
                        Aloss, Bloss, PAB_loss))
        print("A_d_loss_fake: %.2f, A_d_loss_real: %.2f, B_d_loss_fake: %.2f, B_g_loss_real: %.2f, C_d_loss_fake: %.2f, c_d_loss_real: %.2f" \
                    % (Adfake,Adreal,Bdfake,Bdreal, Cdfake, Cdreal))


    def discriminator(self, image,  y=None, prefix='A_', reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name=prefix+'d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(batch_norm(conv2d(h0, self.df_dim*2, name=prefix+'d_h1_conv'), name = prefix+'d_bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*4, name=prefix+'d_h2_conv'), name = prefix+ 'd_bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name=prefix+'d_h3_conv'), name = prefix+ 'd_bn3'))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, prefix+'d_h3_lin')

            return h4
        
    def A_d_net(self, imgs, y = None, reuse = False):
        return self.discriminator(imgs, prefix = 'A_', reuse = reuse)
    
    def B_d_net(self, imgs, y = None, reuse = False):
        return self.discriminator(imgs, prefix = 'B_', reuse = reuse)
        
    def C_d_net(self, imgs, y = None, reuse = False):
        return self.discriminator(imgs, prefix = 'C_', reuse = reuse)
        
    def A_g_net(self, imgs, reuse=False):
        return self.fcn(imgs, prefix='A_g_', reuse = reuse)
        

    def B_g_net(self, imgs, reuse=False):
        return self.fcn(imgs, prefix = 'B_g_', reuse = reuse)
        
    def fcn(self, imgs, prefix=None, reuse = False):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse == False
            
            s = self.image_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # imgs is (256 x 256 x input_c_dim)
            e1 = conv2d(imgs, self.fcn_filter_dim, name=prefix+'e1_conv')
            # e1 is (128 x 128 x self.fcn_filter_dim)
            e2 = batch_norm(conv2d(lrelu(e1), self.fcn_filter_dim*2, name=prefix+'e2_conv'), name = prefix+'bn_e2')
            # e2 is (64 x 64 x self.fcn_filter_dim*2)
            e3 = batch_norm(conv2d(lrelu(e2), self.fcn_filter_dim*4, name=prefix+'e3_conv'), name = prefix+'bn_e3')
            # e3 is (32 x 32 x self.fcn_filter_dim*4)
            e4 = batch_norm(conv2d(lrelu(e3), self.fcn_filter_dim*8, name=prefix+'e4_conv'), name = prefix+'bn_e4')
            # e4 is (16 x 16 x self.fcn_filter_dim*8)
            e5 = batch_norm(conv2d(lrelu(e4), self.fcn_filter_dim*8, name=prefix+'e5_conv'), name = prefix+'bn_e5')
            # e5 is (8 x 8 x self.fcn_filter_dim*8)
            e6 = batch_norm(conv2d(lrelu(e5), self.fcn_filter_dim*8, name=prefix+'e6_conv'), name = prefix+'bn_e6')
            # e6 is (4 x 4 x self.fcn_filter_dim*8)
            e7 = batch_norm(conv2d(lrelu(e6), self.fcn_filter_dim*8, name=prefix+'e7_conv'), name = prefix+'bn_e7')
            # e7 is (2 x 2 x self.fcn_filter_dim*8)
            e8 = batch_norm(conv2d(lrelu(e7), self.fcn_filter_dim*8, name=prefix+'e8_conv'), name = prefix+'bn_e8')
            # e8 is (1 x 1 x self.fcn_filter_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.fcn_filter_dim*8], name=prefix+'d1', with_w=True)
            d1 = tf.nn.dropout(batch_norm(self.d1, name = prefix+'bn_d1'), 0.5)
            if int(self.network_type.split('_')[1]) < 128:
                d1 = tf.concat([d1, e7],3)
            # d1 is (2 x 2 x self.fcn_filter_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.fcn_filter_dim*8], name=prefix+'d2', with_w=True)
            d2 = tf.nn.dropout(batch_norm(self.d2, name = prefix+'bn_d2'), 0.5)
            if int(self.network_type.split('_')[1]) < 64:
                d2 = tf.concat([d2, e6],3)
            # d2 is (4 x 4 x self.fcn_filter_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.fcn_filter_dim*8], name=prefix+'d3', with_w=True)
            d3 = tf.nn.dropout(batch_norm(self.d3, name = prefix+'bn_d3'), 0.5)
            if int(self.network_type.split('_')[1]) < 32:
                d3 = tf.concat([d3, e5],3)
            # d3 is (8 x 8 x self.fcn_filter_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.fcn_filter_dim*8], name=prefix+'d4', with_w=True)
            d4 = batch_norm(self.d4, name = prefix+'bn_d4')
            if int(self.network_type.split('_')[1]) < 16:
                d4 = tf.concat([d4, e4],3)
            # d4 is (16 x 16 x self.fcn_filter_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.fcn_filter_dim*4], name=prefix+'d5', with_w=True)
            d5 = batch_norm(self.d5, name = prefix+'bn_d5')
            if int(self.network_type.split('_')[1]) < 8:
                d5 = tf.concat([d5, e3],3)
            # d5 is (32 x 32 x self.fcn_filter_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.fcn_filter_dim*2], name=prefix+'d6', with_w=True)
            d6 = batch_norm(self.d6, name = prefix+'bn_d6')
            if int(self.network_type.split('_')[1]) < 4:
                d6 = tf.concat([d6, e2],3)
            # d6 is (64 x 64 x self.fcn_filter_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.fcn_filter_dim], name=prefix+'d7', with_w=True)
            d7 = batch_norm(self.d7, name = prefix+'bn_d7')
            if int(self.network_type.split('_')[1]) < 2:
                d7 = tf.concat([d7, e1],3)
            # d7 is (128 x 128 x self.fcn_filter_dim*1*2)

            if prefix == 'B_g_':
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),[self.batch_size, s, s, self.input_channels_A], name=prefix+'d8', with_w=True)
            elif prefix == 'A_g_':
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.input_channels_B], name=prefix+'d8', with_w=True)
             # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)
    
    
    

    def save(self, checkpoint_dir, step):
        model_name = "DualNet.model"
        model_dir = self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir =  self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test DualNet"""
        start_time = time.time()
        tf.global_variables_initializer().run()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
            test_dir = './{}/{}'.format(args.test_dir, self.dir_name)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            test_log = open(test_dir+'evaluation.txt','a') 
            test_log.write(self.dir_name)
            try:
                self.test_domain(args, test_log, type = 'AB')
            except ValueError:
                self.test_domain(args, test_log, type = 'A')
                self.test_domain(args, test_log, type = 'B')
            test_log.close()
        else:
            print(" [!] Load failed...")
        
    def test_domain(self, args, test_log, type = 'A'):
        sample_files = glob('./datasets/{}/val/{}/*.jpg'.format(self.dataset_name,type))
        try:
            n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
            sample_files = [x for (y, x) in sorted(zip(n, sample_files))]
        except:
            try:
                n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0].split('_')[1], sample_files)]
                sample_files = [x for (y, x) in sorted(zip(n, sample_files))]
            except:
                try:
                    n = [int(i) for i in map(lambda x: x.split('/')[-1].split(').jpg')[0].split('(')[0], sample_files)]
                    sample_files = [x for (y, x) in sorted(zip(n, sample_files))]
                except:
                    n = [int(i) for i in map(lambda x: x.split('/')[-1].split(').jpg')[0].split('(')[1], sample_files)]
                    sample_files = [x for (y, x) in sorted(zip(n, sample_files))]
        # load testing input
        print("Loading testing images ...")
        if type != 'AB':
            sample = [load_data(sample_file, is_test=True, image_size =self.image_size, flip = args.flip) for sample_file in sample_files]
        else:
            sample = [load_data_pair(sample_file, img_size =self.image_size) for sample_file in sample_files] 
        sample_images = np.reshape(np.array(sample).astype(np.float32),(len(sample_files),self.image_size, self.image_size,-1))
        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        # test input samples
        if type == 'A':
            aloss_sum = 0.0;
            a_d_loss_sum = 0.0
            b_d_realloss_sum = 0.0
            for i in xrange(0, len(sample_images), self.batch_size):
                idx = i+1
                sample_A_img = np.reshape(np.array(sample_images[i:i+self.batch_size]), (self.batch_size,self.image_size, self.image_size,-1))
                print("sampling A image ", idx)
                translated_A_value, recover_A_value, aloss, a_d_loss, b_d_realloss = self.sess.run(
                    [self.translated_A, self.recover_A, self.A_loss, self.A_d_loss_fake, self.B_d_loss_real],
                    feed_dict={self.real_A: sample_A_img}
                )
                aloss_sum = aloss_sum+ aloss
                a_d_loss_sum = a_d_loss_sum + a_d_loss
                b_d_realloss_sum = b_d_realloss_sum + b_d_realloss
                save_images(sample_A_img, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_real_A.png'.format(args.test_dir, self.dir_name,idx))
                save_images(translated_A_value, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_translated_A.png'.format(args.test_dir, self.dir_name,idx))
                save_images(recover_A_value, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_recover_A.png'.format(args.test_dir, self.dir_name,idx))
            test_log.write('recovery loss of A: %06f \n'%(aloss_sum/sample_images.shape[0]))
            test_log.write('D_A loss of fake:%.2f \n'%(a_d_loss_sum/sample_images.shape[0]))
            test_log.write('D_B loss of real:%.2f \n'%(-b_d_realloss_sum/sample_images.shape[0]))
        elif type=='B':
            bloss_sum = 0.0
            b_d_loss_sum = 0.0
            a_d_realloss_sum = 0.0
            for i in xrange(0, len(sample_images), self.batch_size):
                idx = i+1
                sample_B_img = np.reshape(np.array(sample_images[i:i+self.batch_size]), (self.batch_size,self.image_size, self.image_size,-1))
                print("sampling B image ", idx)

                translated_B_value, recover_B_value,  bloss, b_d_loss, a_d_realloss = self.sess.run(
                    [self.translated_B, self.recover_B, self.B_loss, self.B_d_loss_fake, self.A_d_loss_real],
                    feed_dict={self.real_B:sample_B_img}
                )
                bloss_sum = bloss_sum+ bloss
                b_d_loss_sum = b_d_loss_sum + b_d_loss
                a_d_realloss_sum =a_d_realloss_sum + a_d_realloss

                save_images(sample_B_img, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_real_B.png'.format(args.test_dir, self.dir_name,idx))
                save_images(translated_B_value, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_translated_B.png'.format(args.test_dir, self.dir_name,idx))
                save_images(recover_B_value, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_recover_B.png'.format(args.test_dir, self.dir_name,idx))
            test_log.write('recovery loss of B: %06f\n'%(bloss_sum/sample_images.shape[0]))
            test_log.write('D_B loss of fake:%.2f\n'%(b_d_loss_sum/sample_images.shape[0]))
            test_log.write('D_A loss of real:%.2f\n'%(-a_d_realloss_sum/sample_images.shape[0]))
        elif type == 'AB':
            aloss_sum = a_d_loss_sum = b_d_realloss_sum = 0.0
            bloss_sum = b_d_loss_sum = a_d_realloss_sum = 0.0
            print(sample_images.shape)
            for i in xrange(0, len(sample_images), self.batch_size):
                idx = i+1
                Aimgs = np.array(sample_images[i:i+self.batch_size,:,:,:,0:self.input_channels_A])
                print(Aimgs.shape)
                sample_A_img = np.reshape(Aimgs, (self.batch_size,self.image_size, self.image_size,-1))
                print("sampling A image ", idx)
                translated_A_value, recover_A_value, aloss, a_d_loss, b_d_realloss = self.sess.run(
                    [self.translated_A, self.recover_A, self.A_loss, self.A_d_loss_fake, self.B_d_loss_real],
                    feed_dict={self.real_A: sample_A_img}
                )
                aloss_sum = aloss_sum+ aloss
                a_d_loss_sum = a_d_loss_sum + a_d_loss
                b_d_realloss_sum = b_d_realloss_sum + b_d_realloss
                save_images(sample_A_img, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_real_A.png'.format(args.test_dir, self.dir_name,idx))
                save_images(translated_A_value, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_translated_A.png'.format(args.test_dir, self.dir_name,idx))
                save_images(recover_A_value, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_recover_A.png'.format(args.test_dir, self.dir_name,idx))
            test_log.write('recovery loss of A: %06f \n'%(aloss_sum/sample_images.shape[0]))
            test_log.write('D_A loss of fake:%.2f \n'%(a_d_loss_sum/sample_images.shape[0]))
            test_log.write('D_B loss of real:%.2f \n'%(-b_d_realloss_sum/sample_images.shape[0]))
            
            for i in xrange(0, len(sample_images), self.batch_size):
                idx = i+1
                sample_B_img = np.reshape(np.array(sample_images[i:i+self.batch_size,:,:,:,self.input_channels_A:]), (self.batch_size,self.image_size, self.image_size,-1))
                print("sampling B image ", idx)

                translated_B_value, recover_B_value,  bloss, b_d_loss, a_d_realloss = self.sess.run(
                    [self.translated_B, self.recover_B, self.B_loss, self.B_d_loss_fake, self.A_d_loss_real],
                    feed_dict={self.real_B:sample_B_img}
                )
                bloss_sum = bloss_sum+ bloss
                b_d_loss_sum = b_d_loss_sum + b_d_loss
                a_d_realloss_sum =a_d_realloss_sum + a_d_realloss

                save_images(sample_B_img, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_real_B.png'.format(args.test_dir, self.dir_name,idx))
                save_images(translated_B_value, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_translated_B.png'.format(args.test_dir, self.dir_name,idx))
                save_images(recover_B_value, [self.batch_size, 1],
                            './{}/{}/{:04d}_test_recover_B.png'.format(args.test_dir, self.dir_name,idx))
            test_log.write('recovery loss of B: %06f \n'%(bloss_sum/sample_images.shape[0]))
            test_log.write('D_B loss of fake:%.2f \n'%(b_d_loss_sum/sample_images.shape[0]))
            test_log.write('D_A loss of real:%.2f \n'%(-a_d_realloss_sum/sample_images.shape[0]))