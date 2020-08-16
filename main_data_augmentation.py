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

import data_loader, losses, model
import accuracy

slim = tf.contrib.slim


class CycleGAN:
    """The CycleGAN module."""

    def __init__(self, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore,
                 base_lr, max_step, network_version,
                 csv_name, batch_size, checkpoint_dir, skip, save_freq):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._pool_size = pool_size
        self._size_before_crop = 286
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
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
            (self._pool_size, 64, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_CHANNELS)
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, 64, model.IMG_HEIGHT, model.IMG_WIDTH,
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
                64,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                64,
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

        self.Acc_real_A = accuracy.Acc_real(self.prob_real_a_is_real)
        self.Acc_fake_A = accuracy.Acc_fake(self.prob_fake_a_is_real)
        self.Acc_real_B = accuracy.Acc_real(self.prob_real_b_is_real)
        self.Acc_fake_B = accuracy.Acc_fake(self.prob_fake_b_is_real)

    
    def save_images(self, sess, epoch):
        """
        Saves input and output images.

        :param sess: The session.
        :param epoch: Currnt epoch.
        """
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        names = ['inputA_', 'inputB_', 'fakeB_', 'fakeA_']

        with open(os.path.join(
                self._output_dir, 'epoch_' + str(epoch) + '.html'
        ), 'w') as v_html:
            #for i in range(0, int(np.ceil(self._num_imgs_to_save/self._batch_size))):
            for i in range(self._num_imgs_to_save):
                print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                inputs = sess.run(self.inputs)
                data_csv = read_csv(self._csv_name, header=None, names=['gray','color'], sep=',')
                inputs = {}
                #pdb.set_trace()
                inputs['images_i'] = np.asarray(Image.open(data_csv['gray'][i]))
                inputs['images_i'] = np.expand_dims(inputs['images_i'], axis=0)
                inputs['images_i'] = inputs['images_i'].astype(np.float32)
                inputs['images_i'] = (inputs['images_i']/127.5) - 1

                inputs['images_j'] = np.asarray(Image.open(data_csv['color'][i]))
                inputs['images_j'] = np.expand_dims(inputs['images_j'], axis=0)
                inputs['images_j'] = inputs['images_j'].astype(np.float32)
                inputs['images_j'] = (inputs['images_j']/127.5) - 1

                fake_A_temp, fake_B_temp = sess.run([
                    self.fake_images_a,
                    self.fake_images_b
                ], feed_dict={
                    self.input_a: inputs['images_i'],
                    self.input_b: inputs['images_j']
                })

                tensors = [inputs['images_i'], inputs['images_j'], fake_B_temp, fake_A_temp]

                for name, tensor in zip(names, tensors):                   
                    if 'label_' == name:
                        pass
                        #imsave(os.path.join(self._images_dir, image_name),
                        #       (np.squeeze(tensor[0]) * 255.0).astype(np.uint8))                    
                    elif 'fakeB_' == name:
                        print("shape of fake_B:", fake_B_temp.shape[0])                        
                        for index in np.arange(fake_B_temp.shape[0]):
                            #print("Saving image {}/{}".format(i*self._batch_size + index, self._num_imgs_to_save))
                            image_name = name + str(epoch) + "_" + str(i*self._batch_size + index) + ".png"
                            imsave('./dataset/C%04d.png' %(i*self._batch_size + index), ((tensor[index] + 1) * 127.5).astype(np.uint8))

                    elif 'inputA_' == name:
                        for index in np.arange(inputs['images_i'].shape[0]):
                            image_name = name + str(epoch) + '_%04d.png'
                            image_name = image_name %(i*self._batch_size + index)                             
                            imsave(os.path.join(self._images_dir, image_name),
                                ((np.squeeze(tensor[index]) + 1) * 127.5).astype(np.uint8))
                    v_html.write(
                        "<img src=\"" +
                        os.path.join('imgs', image_name) + "\">"
                    )
                v_html.write("<br>")


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

            self._num_imgs_to_save = 100
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
    pool_size = int(config['pool_size']) if 'pool_size' in config else 50

    to_restore = (to_train == 2)
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 200
    network_version = str(config['network_version'])
    csv_name = str(config['csv_name'])

    save_freq = float(config['_SAVE_FREQ'])

    batch_size = 64

    cyclegan_model = CycleGAN(pool_size, lambda_a, lambda_b, log_dir,
                              to_restore, base_lr, max_step, network_version,
                              csv_name, batch_size, checkpoint_dir, skip, save_freq)

    if to_train > 0:
        cyclegan_model.train()
    else:
        cyclegan_model.test()


if __name__ == '__main__':
    main()