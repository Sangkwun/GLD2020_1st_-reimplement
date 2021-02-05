import os
import math
import argparse
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Create csv to train model')
    parser.add_argument('--csv_path', default='./gldv2_train_total.csv', type=str)
    parser.add_argument('--sample_per_shard', default=800, type=int)
    parser.add_argument('--dataset_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    return parser.parse_args()

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_df(csv_path):
    return pd.read_csv(csv_path)

def main(csv_path, sample_per_shard, dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = get_df(csv_path)

    num_shard = math.ceil(len(df)/sample_per_shard)
    for i in tqdm(range(num_shard)):
        tfrecord_filepath = os.path.join(output_dir, f'gld2020_{str(i).zfill(6)}.tfrecords')        
        start_index = i*sample_per_shard
        end_index = (i+1)*sample_per_shard - 1
        if end_index > len(df):
            end_index = len(df)
        
        with tf.io.TFRecordWriter(path=tfrecord_filepath) as writer:
            for j in range(start_index, end_index):
                sample = df.iloc[j]
                sample_id = sample['id']
                image_path = os.path.join(dataset_dir, sample_id[0], sample_id[1], sample_id[2], f'{sample_id}.jpg')
                if not os.path.exists(image_path):
                    continue
                image_string = open(image_path, 'rb').read()
                image_shape = tf.image.decode_jpeg(image_string).shape

                feature = {
                    'landmark_id': _int64_feature(sample['landmark_id']),
                    'is_clean': _int64_feature(int(sample['is_clean'])),
                    'count': _int64_feature(sample['count']),
                    'loss_weight': _float_feature(sample['loss_weight']),
                    'image': _bytes_feature(image_string),
                    'height': _int64_feature(image_shape[0]),
                    'width': _int64_feature(image_shape[1]),
                    'depth': _int64_feature(image_shape[2]),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

if __name__ == "__main__":
    args = parse_args()
    main(
        args.csv_path,
        args.sample_per_shard,
        args.dataset_dir,
        args.output_dir
    )

'''
python preprocess/get_tfrecord.py \
    --csv_path gldv2_train_all_cls_1061.csv \
    --dataset_dir /Volumes/storage/dataset/luft/google-landmark/train \
    --output_dir ./data/train_all

python preprocess/get_tfrecord.py \
    --csv_path gldv2_train_clean_cls_389.csv \
    --dataset_dir /Volumes/storage/dataset/luft/google-landmark/train \
    --output_dir ./data/train_clean

python preprocess/get_tfrecord.py \
    --csv_path gldv2_val_cls_8682.csv \
    --dataset_dir /Volumes/storage/dataset/luft/google-landmark/train \
    --output_dir ./data/val_all
'''