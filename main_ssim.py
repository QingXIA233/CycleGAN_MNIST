"""Code for training CycleGAN."""
from datetime import datetime
import json
import numpy as np
import os
import random
from scipy.misc import imsave
from matplotlib import pyplot

import click
import tensorflow as tf

import data_loader, loss_ssim, model

slim = tf.contrib.slim


class CycleGAN:
    """The CycleGAN module."""

    def __init__(self, pool_size, lambda_a, lambda_b, lambda_ssim_a, lambda_ssim_b, 
                 output_root_dir, to_restore, base_lr, max_step, network_version,
                 csv_name, batch_size, checkpoint_dir, skip, save_freq):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._pool_size = pool_size
        self._size_before_crop = 286
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._lambda_ssim_a = lambda_ssim_a
        self._lambda_ssim_b = lambda_ssim_b
        self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 1
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._network_version = network_version
        self._csv_name = csv_name
        self._batch_size = batch_size
        self._checkpoint_dir = checkpoint_dir
        self._skip = skip
        self._save_freq = save_freq

        self.fake_images_A = np.zeros(
            (self._pool_size, self._batch_size, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_CHANNELS)
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, self._batch_size, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_CHANNELS)
        )

    def model_setup(self):
        """
        This function sets up the model to train.

        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator
        of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding
        self.fake_A/self.fake_B to corresponding generator.
        This is use to calculate cyclic loss
        """
        self.input_a = tf.placeholder(
            tf.float32, [
                self._batch_size,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                self._batch_size,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="input_B")

        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="fake_pool_B")

        self.global_step = slim.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(
            inputs, network=self._network_version, skip=self._skip)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Various trainer for above loss functions
        *_summ -> Summary variables for above loss functions
        """
        cycle_consistency_loss_a = \
            self._lambda_a * losses.cycle_consistency_loss(
                real_images=self.input_a, generated_images=self.cycle_images_a,
            )
        cycle_consistency_loss_b = \
            self._lambda_b * losses.cycle_consistency_loss(
                real_images=self.input_b, generated_images=self.cycle_images_b,
            )

        cycle_consistency_ssim_loss_a = \
            self._lambda_ssim_a * losses.cycle_consistency_ssim_loss(
                real_images=self.input_a, generated_images=self.cycle_images_a,
            )
        cycle_consistency_ssim_loss_b = \
            self._lambda_ssim_b * losses.cycle_consistency_ssim_loss(
                real_images=self.input_b, generated_images=self.cycle_images_b,
            )

        lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)

        lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)

        g_loss_A = \
            cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b + cycle_consistency_ssim_loss_a + cycle_consistency_ssim_loss_b
        g_loss_B = \
            cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a + cycle_consistency_ssim_loss_b + cycle_consistency_ssim_loss_a

        d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )
        d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        self.g_A_loss = g_loss_A
        self.g_B_loss = g_loss_B
        self.d_A_loss = d_loss_A
        self.d_B_loss = d_loss_B

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    def save_images(self, sess, epoch):
        """
        Saves input and output images.

        :param sess: The session.
        :param epoch: Currnt epoch.
        """
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        names = ['inputA_', 'inputB_', 'fakeA_',
                 'fakeB_', 'cycA_', 'cycB_']

        with open(os.path.join(
                self._output_dir, 'epoch_' + str(epoch) + '.html'
        ), 'w') as v_html:
            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                inputs = sess.run(self.inputs)
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                    self.fake_images_a,
                    self.fake_images_b,
                    self.cycle_images_a,
                    self.cycle_images_b
                ], feed_dict={
                    self.input_a: inputs['images_i'],
                    self.input_b: inputs['images_j']
                })

                tensors = [inputs['images_i'], inputs['images_j'],
                           fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

                for name, tensor in zip(names, tensors):
                    image_name = name + str(epoch) + "_" + str(i) + ".png" 
                    imsave(os.path.join(self._images_dir, image_name),
                           ((tensor[0] + 1) * 127.5).astype(np.uint8)
                           )
                    v_html.write(
                        "<img src=\"" +
                        os.path.join('imgs', image_name) + "\">"
                    )
                v_html.write("<br>")


    def fake_image_pool(self, num_fakes, fake, fake_pool):
        """
        This function saves the generated image to corresponding
        pool of images.

        It keeps on feeling the pool till it is full and then randomly
        selects an already stored image and replace it with new one.
        """
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

     # create a line plot of loss for the gan and save to file
    def plot_loss_history(self, g_A_hist, g_B_hist, d_A_hist, d_B_hist):
        # plot losses for mapping A to B
        pyplot.subplot(2, 1, 1)
        pyplot.plot(g_A_hist, label='g_A_loss')
        pyplot.plot(d_B_hist, label='d_B_loss')
        pyplot.legend()
        # plot discriminator accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.plot(g_B_hist, label='g_B_loss')
        pyplot.plot(d_A_hist, label='d_A_loss')
        pyplot.legend()
        # save plot to file
        pyplot.savefig('./output/cyclegan/plot_line_plot_loss.png')
        pyplot.close()

     # create a line plot of loss for the gan and save to file
    def plot_acc_history(self, Acc_real_A_hist, Acc_real_B_hist, Acc_fake_A_hist, Acc_fake_B_hist):
        # plot losses for mapping A to B
        pyplot.subplot(2, 1, 1)
        pyplot.plot(Acc_real_A_hist, label='acc_real_A')
        pyplot.plot(Acc_fake_A_hist, label='acc_fake_A')
        pyplot.legend()
        # plot discriminator accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.plot(Acc_real_B_hist, label='acc_real_B')
        pyplot.plot(Acc_fake_B_hist, label='acc_fake_B')
        pyplot.legend()
        # save plot to file
        pyplot.savefig('./output/cyclegan/plot_line_plot_accuracy.png')
        pyplot.close()

    def train(self):
        """Training Function."""
        # Load Dataset from the dataset folder
        self.inputs = data_loader.load_data(self._csv_name, self._batch_size)

        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver()

        max_images = 10000

        with tf.Session() as sess:
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # prepare lists for storing loss stats each iteration
            g_A_hist, g_B_hist, d_A_hist, d_B_hist = list(), list(), list(), list()

            # prepare lists for storing accuracy stats each iteration
            """acc_real_A_hist, acc_real_B_hist, acc_fake_A_hist, acc_fake_B_hist = list(), list(), list(), list()"""

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step):
                print("In the epoch ", epoch)

                if (epoch+1) % self._save_freq == 0:
                    saver.save(sess, os.path.join(
                        self._output_dir, "cyclegan"), global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                if epoch < 100:
                    curr_lr = self._base_lr
                else:
                    curr_lr = self._base_lr - \
                        self._base_lr * (epoch - 100) / 100

                if (epoch+1) % self._save_freq == 0:
                    self.save_images(sess, epoch)

                for i in range(0, max_images // 64):
                    print("Processing batch {}/{}".format(i, max_images // 64))

                    inputs = sess.run(self.inputs)

                    # Optimizing the G_A network
                    _, fake_B_temp, g_A_loss, summary_str = sess.run(
                        [self.g_A_trainer,
                         self.fake_images_b,
                         self.g_A_loss,
                         self.g_A_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images // 64 + i)

                    fake_B_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimizing the D_B network
                    _, d_B_loss, summary_str = sess.run(
                        [self.d_B_trainer, self.d_B_loss, self.d_B_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_B: fake_B_temp1
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images // 64 + i)

                    # Optimizing the G_B network
                    _, fake_A_temp, g_B_loss, summary_str = sess.run(
                        [self.g_B_trainer,
                         self.fake_images_a,
                         self.g_B_loss,
                         self.g_B_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images // 64 + i)

                    fake_A_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, d_A_loss, summary_str = sess.run(
                        [self.d_A_trainer, self.d_A_loss, self.d_A_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_A: fake_A_temp1
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images // 64 + i)

                    # summarize the losses on this batch

                    print('>%d, g_A=%.3f, g_B=%.3f, d_A=%.3f, d_B=%.3f ' %
                        (i+1, g_A_loss, g_B_loss, d_A_loss, d_B_loss))

                    g_A_hist.append(g_A_loss)
                    g_B_hist.append(g_B_loss)
                    d_A_hist.append(d_A_loss)
                    d_B_hist.append(d_B_loss)                 

                    '''print('>%d, acc_real_A=%.3f, acc_real_B=%.3f, acc_fake_A=%.3f, acc_fake_B=%.3f ' %
                        (i+1, acc_real_A, acc_real_B, acc_fake_A, acc_fake_B))'''

                    writer.flush()
                    self.num_fake_inputs += 64

                sess.run(tf.assign(self.global_step, epoch + 1))
                self.plot_loss_history(g_A_hist, g_B_hist, d_A_hist, d_B_hist)
                '''self.plot_acc_history(acc_real_A_hist, acc_real_B_hist, acc_fake_A_hist, acc_fake_B_hist)'''


            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)

    def test(self):
        """Test Function."""
        print("Testing the results")

        self.inputs = data_loader.load_data(self._csv_name, self._batch_size)

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
            saver.restore(sess, chkpt_fname)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self._num_imgs_to_save = 10000
            self.save_images(sess, 0)

            coord.request_stop()
            coord.join(threads)


@click.command()
@click.option('--to_train',
              type=click.INT,
              default=True,
              help='Whether it is train or false.')
@click.option('--log_dir',
              type=click.STRING,
              default=None,
              help='Where the data is logged to.')
@click.option('--config_filename',
              type=click.STRING,
              default='train',
              help='The name of the configuration file.')
@click.option('--checkpoint_dir',
              type=click.STRING,
              default='',
              help='The name of the train/test split.')
@click.option('--skip',
              type=click.BOOL,
              default=False,
              help='Whether to add skip connection between input and output.')
def main(to_train, log_dir, config_filename, checkpoint_dir, skip):
    """

    :param to_train: Specify whether it is training or testing. 1: training; 2:
     resuming from latest checkpoint; 0: testing.
    :param log_dir: The root dir to save checkpoints and imgs. The actual dir
    is the root dir appended by the folder with the name timestamp.
    :param config_filename: The configuration file.
    :param checkpoint_dir: The directory that saves the latest checkpoint. It
    only takes effect when to_train == 2.
    :param skip: A boolean indicating whether to add skip connection between
    input and output.
    """
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    lambda_a = float(config['_LAMBDA_A']) if '_LAMBDA_A' in config else 10.0
    lambda_b = float(config['_LAMBDA_B']) if '_LAMBDA_B' in config else 10.0
    lambda_ssim_a = float(config['_LAMBDA_SSIM_A']) if '_LAMBDA_SSIM_A' in config else 10.0
    lambda_ssim_b = float(config['_LAMBDA_SSI8_B']) if '_LAMBDA_SSIM_B' in config else 10.0
    pool_size = int(config['pool_size']) if 'pool_size' in config else 50

    to_restore = (to_train == 2)
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 200
    network_version = str(config['network_version'])
    csv_name = str(config['csv_name'])

    save_freq = float(config['_SAVE_FREQ'])

    batch_size = 64

    cyclegan_model = CycleGAN(pool_size, lambda_a, lambda_b, lambda_ssim_a, lambda_ssim_b, 
                              log_dir, to_restore, base_lr, max_step, network_version,
                              csv_name, batch_size, checkpoint_dir, skip, save_freq)

    if to_train > 0:
        cyclegan_model.train()
    else:
        cyclegan_model.test()


if __name__ == '__main__':
    main()
