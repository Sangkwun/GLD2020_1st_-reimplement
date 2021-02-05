import os
import json
import logging
import argparse

import tensorflow as tf

from functools import partial
from model import build_model

AUTOTUNE = tf.data.experimental.AUTOTUNE

def parse_args():
    parser = argparse.ArgumentParser(description='Create csv to train model')
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--config_path', required=True, type=str)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, mode='r', encoding='utf-8') as f:
        configs = json.loads(f.read())
    return configs

def read_tfrecord(example, image_size):
    image_feature_description = {
        'landmark_id': tf.io.FixedLenFeature([], tf.int64),
        'is_clean': tf.io.FixedLenFeature([], tf.int64),
        'count': tf.io.FixedLenFeature([], tf.int64),
        'loss_weight': tf.io.FixedLenFeature([], tf.float32),
        'image': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, features=image_feature_description)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, image_size)
    landmark_id = tf.cast(example["landmark_id"], tf.int32)
    loss_weight = tf.cast(example['loss_weight'], tf.float32)
    return image,  landmark_id, loss_weight

def load_dataset(data_dir, image_size=[512,512], batch_size=16):
    tf_record_filenames = tf.io.gfile.glob(os.path.join(data_dir, '*.tfrecords'))
    data_option = tf.data.Options()
    data_option.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(tf_record_filenames)
    dataset = dataset.with_options(data_option)
    dataset = dataset.map(
        partial(read_tfrecord, image_size=image_size), num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def main(exp_name, config_path):
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, exp_name), exist_ok=True)
    configs = load_config(config_path)

    image_size = configs['image_size']
    num_classes = configs['num_classes']
    batch_size = configs['batch_size']

    dataset = load_dataset(configs['data_dir'], image_size=image_size, batch_size=batch_size)
    strategy = tf.distribute.get_strategy()
    with strategy.scope():

        model = build_model(
            'efficient-b5',
            image_size,
            num_classes
        )

    model.fit(
        dataset,
        epochs=1,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.exp_name, args.config_path)
"""
python train.py \
    --exp_name 20200204_debug \
    --config_path ./configs/train_debug.json

"""