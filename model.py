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
    def __init__(self, sess, image_size=256, batch_size=1,fcn_filter_dim = 64,  \
                 A_channels = 3, B_channels = 3, dataset_name='facades', \
                 checkpoint_dir=None, lambda_A = 20., lambda_B = 20., \
                 sample_dir=None, loss_metric = 'L1', flip = False):
        self.df_dim = fcn_filter_dim
        self.flip = flip
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        
        self.sess = sess
        self.is_grayscale_A = (A_channels == 1)
        self.is_grayscale_B = (B_channels == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.fcn_filter_dim = fcn_filter_dim
        self.A_channels = A_channels
        self.B_channels = B_channels
        self.loss_metric = loss_metric

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        
        #directory name for output and logs saving
        self.dir_name = "%s-img_sz_%s-fltr_dim_%d-%s-lambda_AB_%s_%s" % (
                    self.dataset_name, 
                    self.image_size,
                    self.fcn_filter_dim,
                    self.loss_metric, 
                    self.lambda_A, 
                    self.lambda_B
                ) 
        self.build_model()

    def build_model(self):
    ###    define place holders
        self.real_A = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size,
                                         self.A_channels ],name='real_A')
        self.real_B = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size,
                                         self.B_channels ], name='real_B')
        
    ###  define graphs
        self.A2B = self.A_g_net(self.real_A, reuse = False)
        self.B2A = self.B_g_net(self.real_B, reuse = False)
        self.A2B2A = self.B_g_net(self.A2B, reuse = True)
        self.B2A2B = self.A_g_net(self.B2A, reuse = True)
        
        if self.loss_metric == 'L1':
            self.A_loss = tf.reduce_mean(tf.abs(self.A2B2A - self.real_A))
            self.B_loss = tf.reduce_mean(tf.abs(self.B2A2B - self.real_B))
        elif self.loss_metric == 'L2':
            self.A_loss = tf.reduce_mean(tf.square(self.A2B2A - self.real_A))
            self.B_loss = tf.reduce_mean(tf.square(self.B2A2B - self.real_B))
        
        self.Ad_logits_fake = self.A_d_net(self.A2B, reuse = False)
        self.Ad_logits_real = self.A_d_net(self.real_B, reuse = True)
        self.Ad_loss_real = celoss(self.Ad_logits_real, tf.ones_like(self.Ad_logits_real))
        self.Ad_loss_fake = celoss(self.Ad_logits_fake, tf.zeros_like(self.Ad_logits_fake))
        self.Ad_loss = self.Ad_loss_fake + self.Ad_loss_real
        self.Ag_loss = celoss(self.Ad_logits_fake, labels=tf.ones_like(self.Ad_logits_fake))+self.lambda_B * (self.B_loss )

        self.Bd_logits_fake = self.B_d_net(self.B2A, reuse = False)
        self.Bd_logits_real = self.B_d_net(self.real_A, reuse = True)
        self.Bd_loss_real = celoss(self.Bd_logits_real, tf.ones_like(self.Bd_logits_real))
        self.Bd_loss_fake = celoss(self.Bd_logits_fake, tf.zeros_like(self.Bd_logits_fake))
        self.Bd_loss = self.Bd_loss_fake + self.Bd_loss_real
        self.Bg_loss = celoss(self.Bd_logits_fake, tf.ones_like(self.Bd_logits_fake))+self.lambda_A * (self.A_loss)
       
        self.d_loss = self.Ad_loss + self.Bd_loss
        self.g_loss = self.Ag_loss + self.Bg_loss
        ## define trainable variables
        t_vars = tf.trainable_variables()
        self.A_d_vars = [var for var in t_vars if 'A_d_' in var.name]
        self.B_d_vars = [var for var in t_vars if 'B_d_' in var.name]
        self.A_g_vars = [var for var in t_vars if 'A_g_' in var.name]
        self.B_g_vars = [var for var in t_vars if 'B_g_' in var.name]
        self.d_vars = self.A_d_vars + self.B_d_vars 
        self.g_vars = self.A_g_vars + self.B_g_vars
        self.saver = tf.train.Saver()

    def clip_trainable_vars(self, var_list):
        for var in var_list:
            self.sess.run(var.assign(tf.clip_by_value(var, -self.c, self.c)))

    def load_random_samples(self):
        #np.random.choice(
        sample_files =np.random.choice(glob('./datasets/{}/val/A/*.jpg'.format(self.dataset_name)),self.batch_size)
        sample_A_imgs = [load_data(f, image_size =self.image_size, flip = False) for f in sample_files]
        
        sample_files = np.random.choice(glob('./datasets/{}/val/B/*.jpg'.format(self.dataset_name)),self.batch_size)
        sample_B_imgs = [load_data(f, image_size =self.image_size, flip = False) for f in sample_files]

        sample_A_imgs = np.reshape(np.array(sample_A_imgs).astype(np.float32),(self.batch_size,self.image_size, self.image_size,-1))
        sample_B_imgs = np.reshape(np.array(sample_B_imgs).astype(np.float32),(self.batch_size,self.image_size, self.image_size,-1))
        return sample_A_imgs, sample_B_imgs

    def sample_shotcut(self, sample_dir, epoch_idx, batch_idx):
        sample_A_imgs,sample_B_imgs = self.load_random_samples()
        
        Ag, A2B2A_imgs, A2B_imgs = self.sess.run([self.A_loss, self.A2B2A, self.A2B], feed_dict={self.real_A: sample_A_imgs, self.real_B: sample_B_imgs})
        Bg, B2A2B_imgs, B2A_imgs = self.sess.run([self.B_loss, self.B2A2B, self.B2A], feed_dict={self.real_A: sample_A_imgs, self.real_B: sample_B_imgs})

        save_images(A2B_imgs, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_A2B.jpg'.format(sample_dir,self.dir_name , epoch_idx, batch_idx))
        save_images(A2B2A_imgs, [self.batch_size,1],    './{}/{}/{:06d}_{:04d}_A2B2A.jpg'.format(sample_dir,self.dir_name, epoch_idx,  batch_idx))
        
        save_images(B2A_imgs, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_B2A.jpg'.format(sample_dir,self.dir_name, epoch_idx, batch_idx))
        save_images(B2A2B_imgs, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_B2A2B.jpg'.format(sample_dir,self.dir_name, epoch_idx, batch_idx))
        
        print("[Sample] A_loss: {:.8f}, B_loss: {:.8f}".format(Ag, Bg))

    def train(self, args):
        """Train Dual GAN"""
        decay = 0.9
        self.d_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay) \
                          .minimize(self.d_loss, var_list=self.d_vars)
                          
        self.g_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay) \
                          .minimize(self.g_loss, var_list=self.g_vars)          
        tf.global_variables_initializer().run()

        self.writer = tf.summary.FileWriter("./logs/"+self.dir_name, self.sess.graph)

        step = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" Load failed...ignored...")
            print(" start training...")

        for epoch_idx in xrange(args.epoch):
            data_A = glob('./datasets/{}/train/A/*.jpg'.format(self.dataset_name))
            data_B = glob('./datasets/{}/train/B/*.jpg'.format(self.dataset_name))
            np.random.shuffle(data_A)
            np.random.shuffle(data_B)
            epoch_size = min(len(data_A), len(data_B)) // (self.batch_size)
            print('[*] training data loaded successfully')
            print("#data_A: %d  #data_B:%d" %(len(data_A),len(data_B)))
            print('[*] run optimizor...')

            for batch_idx in xrange(0, epoch_size):
                imgA_batch = self.load_training_imgs(data_A, batch_idx)
                imgB_batch = self.load_training_imgs(data_B, batch_idx)
                
                print("Epoch: [%2d] [%4d/%4d]"%(epoch_idx, batch_idx, epoch_size))
                step = step + 1
                self.run_optim(imgA_batch, imgB_batch, step, start_time)

                if np.mod(step, 100) == 1:
                    self.sample_shotcut(args.sample_dir, epoch_idx, batch_idx)

                if np.mod(step, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, step)

    def load_training_imgs(self, files, idx):
        batch_files = files[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_imgs = [load_data(f, image_size =self.image_size, flip = self.flip) for f in batch_files]
                
        batch_imgs = np.reshape(np.array(batch_imgs).astype(np.float32),(self.batch_size,self.image_size, self.image_size,-1))
        
        return batch_imgs
        
    def run_optim(self,batch_A_imgs, batch_B_imgs,  counter, start_time):
        _, Adfake,Adreal,Bdfake,Bdreal, Ad, Bd = self.sess.run(
            [self.d_optim, self.Ad_loss_fake, self.Ad_loss_real, self.Bd_loss_fake, self.Bd_loss_real, self.Ad_loss, self.Bd_loss], 
            feed_dict = {self.real_A: batch_A_imgs, self.real_B: batch_B_imgs})
        _, Ag, Bg, Aloss, Bloss = self.sess.run(
            [self.g_optim, self.Ag_loss, self.Bg_loss, self.A_loss, self.B_loss], 
            feed_dict={ self.real_A: batch_A_imgs, self.real_B: batch_B_imgs})

        _, Ag, Bg, Aloss, Bloss = self.sess.run(
            [self.g_optim, self.Ag_loss, self.Bg_loss, self.A_loss, self.B_loss], 
            feed_dict={ self.real_A: batch_A_imgs, self.real_B: batch_B_imgs})

        print("time: %4.4f, Ad: %.2f, Ag: %.2f, Bd: %.2f, Bg: %.2f,  U_diff: %.5f, V_diff: %.5f" \
                    % (time.time() - start_time, Ad,Ag,Bd,Bg, Aloss, Bloss))
        print("Ad_fake: %.2f, Ad_real: %.2f, Bd_fake: %.2f, Bg_real: %.2f" % (Adfake,Adreal,Bdfake,Bdreal))

    def A_d_net(self, imgs, y = None, reuse = False):
        return self.discriminator(imgs, prefix = 'A_d_', reuse = reuse)
    
    def B_d_net(self, imgs, y = None, reuse = False):
        return self.discriminator(imgs, prefix = 'B_d_', reuse = reuse)
        
    def discriminator(self, image,  y=None, prefix='A_d_', reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name=prefix+'h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(batch_norm(conv2d(h0, self.df_dim*2, name=prefix+'h1_conv'), name = prefix+'bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*4, name=prefix+'h2_conv'), name = prefix+ 'bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name=prefix+'h3_conv'), name = prefix+ 'bn3'))
            # h3 is (32 x 32 x self.df_dim*8)
            h4 = conv2d(h3, 1, d_h=1, d_w=1, name =prefix+'h4')
            return h4
        
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
            d1 = tf.concat([d1, e7],3)
            # d1 is (2 x 2 x self.fcn_filter_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.fcn_filter_dim*8], name=prefix+'d2', with_w=True)
            d2 = tf.nn.dropout(batch_norm(self.d2, name = prefix+'bn_d2'), 0.5)

            d2 = tf.concat([d2, e6],3)
            # d2 is (4 x 4 x self.fcn_filter_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.fcn_filter_dim*8], name=prefix+'d3', with_w=True)
            d3 = tf.nn.dropout(batch_norm(self.d3, name = prefix+'bn_d3'), 0.5)

            d3 = tf.concat([d3, e5],3)
            # d3 is (8 x 8 x self.fcn_filter_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.fcn_filter_dim*8], name=prefix+'d4', with_w=True)
            d4 = batch_norm(self.d4, name = prefix+'bn_d4')

            d4 = tf.concat([d4, e4],3)
            # d4 is (16 x 16 x self.fcn_filter_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.fcn_filter_dim*4], name=prefix+'d5', with_w=True)
            d5 = batch_norm(self.d5, name = prefix+'bn_d5')
            d5 = tf.concat([d5, e3],3)
            # d5 is (32 x 32 x self.fcn_filter_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.fcn_filter_dim*2], name=prefix+'d6', with_w=True)
            d6 = batch_norm(self.d6, name = prefix+'bn_d6')
            d6 = tf.concat([d6, e2],3)
            # d6 is (64 x 64 x self.fcn_filter_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.fcn_filter_dim], name=prefix+'d7', with_w=True)
            d7 = batch_norm(self.d7, name = prefix+'bn_d7')
            d7 = tf.concat([d7, e1],3)
            # d7 is (128 x 128 x self.fcn_filter_dim*1*2)

            if prefix == 'B_g_':
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),[self.batch_size, s, s, self.A_channels], name=prefix+'d8', with_w=True)
            elif prefix == 'A_g_':
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),[self.batch_size, s, s, self.B_channels], name=prefix+'d8', with_w=True)
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
            self.test_domain(args, test_log, type = 'A')
            self.test_domain(args, test_log, type = 'B')
            test_log.close()
        
    def test_domain(self, args, test_log, type = 'A'):
        test_files = glob('./datasets/{}/val/{}/*.jpg'.format(self.dataset_name,type))
        # load testing input
        print("Loading testing images ...")
        test_imgs = [load_data(f, is_test=True, image_size =self.image_size, flip = args.flip) for f in test_files]
        print("#images loaded: %d"%(len(test_imgs)))
        test_imgs = np.reshape(np.asarray(test_imgs).astype(np.float32),(len(test_files),self.image_size, self.image_size,-1))
        test_imgs = [test_imgs[i*self.batch_size:(i+1)*self.batch_size]
                         for i in xrange(0, len(test_imgs)//self.batch_size)]
        test_imgs = np.asarray(test_imgs)
        test_path = './{}/{}/'.format(args.test_dir, self.dir_name)
        # test input samples
        if type == 'A':
            for i in xrange(0, len(test_files)//self.batch_size):
                filename_o = test_files[i*self.batch_size].split('/')[-1].split('.')[0]
                print(filename_o)
                idx = i+1
                A_imgs = np.reshape(np.array(test_imgs[i]), (self.batch_size,self.image_size, self.image_size,-1))
                print("testing A image %d"%(idx))
                print(A_imgs.shape)
                A2B_imgs, A2B2A_imgs = self.sess.run(
                    [self.A2B, self.A2B2A],
                    feed_dict={self.real_A: A_imgs}
                    )
                save_images(A_imgs, [self.batch_size, 1], test_path+filename_o+'_realA.jpg')
                save_images(A2B_imgs, [self.batch_size, 1], test_path+filename_o+'_A2B.jpg')
                save_images(A2B2A_imgs, [self.batch_size, 1], test_path+filename_o+'_A2B2A.jpg')
        elif type=='B':
            for i in xrange(0, len(test_files)//self.batch_size):
                filename_o = test_files[i*self.batch_size].split('/')[-1].split('.')[0]
                idx = i+1
                B_imgs = np.reshape(np.array(test_imgs[i]), (self.batch_size,self.image_size, self.image_size,-1))
                print("testing B image %d"%(idx))
                B2A_imgs, B2A2B_imgs = self.sess.run(
                    [self.B2A, self.B2A2B],
                    feed_dict={self.real_B:B_imgs}
                    )
                save_images(B_imgs, [self.batch_size, 1],test_path+filename_o+'_realB.jpg')
                save_images(B2A_imgs, [self.batch_size, 1],test_path+filename_o+'_B2A.jpg')
                save_images(B2A2B_imgs, [self.batch_size, 1],test_path+filename_o+'_B2A2B.jpg')