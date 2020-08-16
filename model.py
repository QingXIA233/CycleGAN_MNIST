"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf
import layers

# The number of samples per batch.
BATCH_SIZE = 64

# The height of each image.
IMG_HEIGHT = 28

# The width of each image.
IMG_WIDTH = 28

# The number of color channels per image.
IMG_CHANNELS = 3

POOL_SIZE = 50
ngf = 32
ndf = 64


def get_outputs(inputs, network="tensorflow", skip=False):
    images_a = inputs['images_a']
    images_b = inputs['images_b']

    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']

    with tf.variable_scope("Model") as scope:

        current_discriminator = patch_discriminator
        current_generator = build_generator_resnet_9blocks

        prob_real_a_is_real = current_discriminator(images_a, "d_A")
        prob_real_b_is_real = current_discriminator(images_b, "d_B")

        fake_images_b = current_generator(images_a, name="g_A", skip=skip)
        fake_images_a = current_generator(images_b, name="g_B", skip=skip)

        scope.reuse_variables()

        prob_fake_a_is_real = current_discriminator(fake_images_a, "d_A")
        prob_fake_b_is_real = current_discriminator(fake_images_b, "d_B")

        cycle_images_a = current_generator(fake_images_b, "g_B", skip=skip)
        cycle_images_b = current_generator(fake_images_a, "g_A", skip=skip)

        scope.reuse_variables()

        prob_fake_pool_a_is_real = current_discriminator(fake_pool_a, "d_A")
        prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, "d_B")

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
    }


def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    """build a single block of resnet.

    :param inputres: inputres
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)


def build_generator_resnet_9blocks(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        #pad_input.shape = (1, 134, 134, 3)
        o_c1 = layers.general_conv2d(
            pad_input, ngf, f, f, 1, 1, 0.02, name="c1")
        #o_c1.shape = (1, 128, 128, 32)
        o_c2 = layers.general_conv2d(
            o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        #o_c2.shape = (1, 64, 64, 64)
        o_c3 = layers.general_conv2d(
            o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")
        #o_c3.shape = (1, 32, 32, 128)

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding) 
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)
        #o_r9.shape = (1, 32, 32, 128)

        o_d1 = layers.general_deconv2d(
            o_r9, [BATCH_SIZE, 14, 14, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
            "SAME", "d1")
        #o_c5.shape = (1, 64, 64, 64)

        o_d2 = layers.general_deconv2d(
            o_d1, [BATCH_SIZE, 28, 28, ngf], ngf, ks, ks, 2, 2, 0.02,
            "SAME", "d2")
        #o_c5.shape = (1, 128, 128, 32)

        o_c5 = layers.general_conv2d(o_d2, IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c5",
                                     do_norm=False, do_relu=False)
        #o_c6.shape = (1, 128, 128, 3)

        if skip is True:
            out_gen = tf.nn.tanh(inputgen + o_c5, "t1")
        else:
            out_gen = tf.nn.tanh(o_c5, "t1")

        return out_gen


def patch_discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        patch_input = tf.random_crop(inputdisc, [1, 7, 7, 3])
        o_c1 = layers.general_conv2d(patch_input, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm="False",
                                     relufactor=0.2)
        o_c2 = layers.general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv2d(o_c3, ndf * 8, f, f, 2, 2,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv2d(
            o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False,
            do_relu=False)

        return o_c5