"""Create csv files for loading the training and testing data."""
import csv
import os
import random

import click

def create_list(foldername, fulldir=True, suffix=".png"):
    """
    :param foldername: The full path of the folder.
    :param fulldir: Whether to return the full path or not.
    :param suffix: Filter by suffix.
    :return: The list of filenames in the folder with given suffix.
    """
    file_list_tmp = os.listdir(foldername)
    file_list = []
    if fulldir:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    return file_list

@click.command()
@click.option('--image_path_a',
              type=click.STRING,
              default='./input/train/gray',
              help='The path to the images from original MNIST datasets.')
@click.option('--image_path_b',
              type=click.STRING,
              default='./input/train/RGB',
              help='The path to the images from colored-digits MNIST.')
@click.option('--output_path',
              type=click.STRING,
              default='./input/train/mnist_train.csv',
              help='The path to the corresponding csv file for the input data.')

def create_csv(image_path_a, image_path_b, output_path):
    list_img_a = create_list(image_path_a, True, suffix=".png")
    list_img_b = create_list(image_path_b, True, suffix=".png")

    num_rows = 10000
    all_data_tuples = []
    for i in range(num_rows):
        all_data_tuples.append((
            list_img_a[i % len(list_img_a)],
            list_img_b[i % len(list_img_b)]
        ))
        
    with open(output_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for data_tuple in enumerate(all_data_tuples):
            csv_writer.writerow(list(data_tuple[1]))


if __name__ == '__main__':
    create_csv()