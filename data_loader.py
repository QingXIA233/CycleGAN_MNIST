import tensorflow as tf
import model

def load_samples(csv_name):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)

    image_decoded_A = tf.image.decode_png(
            file_contents_i, channels=model.IMG_CHANNELS, dtype=tf.uint8)
    image_decoded_B = tf.image.decode_png(
            file_contents_j, channels=model.IMG_CHANNELS, dtype=tf.uint8)

    image_decoded_A = tf.cast(image_decoded_A, tf.float32)
    image_decoded_B = tf.cast(image_decoded_B, tf.float32)

    return image_decoded_A, image_decoded_B

def load_data(csv_name, batch_size):

    image_i, image_j = load_samples(csv_name)

    # Preprocessing:
    image_i = tf.image.resize_images(
        image_i, [model.IMG_HEIGHT, model.IMG_WIDTH])
    image_j = tf.image.resize_images(
        image_j, [model.IMG_HEIGHT, model.IMG_WIDTH])

    image_i = tf.subtract(tf.div(image_i, 127.5), 1)
    image_j = tf.subtract(tf.div(image_j, 127.5), 1)

    images_i, images_j = tf.train.shuffle_batch([image_i, image_j], batch_size, 5000, 100)

    inputs = {
        'images_i': images_i,
        'images_j': images_j
        }

    return inputs

