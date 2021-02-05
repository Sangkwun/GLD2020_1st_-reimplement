import random
import swifter
import argparse
import pandas as pd
import numpy as np
import collections
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Create csv to train model')
    parser.add_argument('--total_csv_path', required=True, type=str)
    parser.add_argument('--cleaned_csv_path', required=True, type=str)
    return parser.parse_args()

def add_landmark_count(images):
    return len(images.split(' '))

def add_attributes(row, clean_train, landmark_id_count):
    _id = row['id']
    _landmark_id = row['landmark_id']
    landmark_clean_series = clean_train[clean_train['landmark_id']==_landmark_id]
    row['is_clean'] = _id in landmark_clean_series['images'].values[0] if len(landmark_clean_series['images'].values) > 0 else False
    row['count'] = landmark_id_count[_landmark_id]
    return row

def add_loss_weight(row, landmark_id_loss_weight=None):
    _landmark_id = row['landmark_id']
    if landmark_id_loss_weight is None:
        row['loss_weight'] = 1
    else:
        row['loss_weight'] = landmark_id_loss_weight[_landmark_id]
    return row

def get_loss_weight(landmark_id_count, total_sample):
    loss_weight = dict()
    for landmark_id, count in landmark_id_count.items():
        log_scale = np.log(total_sample/count) # balanced weight
        loss_weight[landmark_id] = log_scale
    return loss_weight

def split_val_set(df, counter):
    valid_id_list = [k for k, v in counter.items() if v >= 4]
    df['val'] = False

    for _id in tqdm(valid_id_list):
        _samples = df[df['landmark_id']==_id]
        if len(_samples) == 0:
            continue
        _index = random.choice(_samples.index)
        df.iloc[_index, df.columns.get_loc('val')] = True
    return df

def main(total_csv_path, cleaned_csv_path):
    total_train = pd.read_csv(total_csv_path)
    clean_train = pd.read_csv(cleaned_csv_path)

    landmark_id_count = collections.Counter(total_train['landmark_id'].tolist())
    total_train = total_train[:10000]
    total_set = split_val_set(total_train, landmark_id_count)
    del total_set['url']

    total_set = total_set.swifter.apply(add_attributes, axis=1, landmark_id_count=landmark_id_count, clean_train=clean_train)
    train_set = total_set[~total_set['val']]
    val_set = total_set[total_set['val']]
    del total_set

    landmark_id_count = collections.Counter(total_train['landmark_id'].tolist())
    train_set_loss_weight = get_loss_weight(landmark_id_count, len(landmark_id_count))
    train_set = train_set.swifter.apply(add_loss_weight, axis=1, landmark_id_loss_weight=train_set_loss_weight)
    val_set = val_set.swifter.apply(add_loss_weight, axis=1)
    
    train_set_classes = len(set(train_set['landmark_id'].tolist()))
    val_set_classes = len(set(val_set['landmark_id'].tolist()))
    train_set.to_csv(f'./gldv2_train_all_cls_{train_set_classes}.csv', index=False)
    val_set.to_csv(f'./gldv2_val_cls_{val_set_classes}.csv', index=False)

    train_clean_set = train_set[train_set['is_clean']]
    train_clean_set_classes = len(set(train_clean_set['landmark_id'].tolist()))
    train_clean_set.to_csv(f'./gldv2_train_clean_cls_{train_clean_set_classes}.csv', index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args.total_csv_path, args.cleaned_csv_path)

"""
python preprocess/get_csv.py \
    --total_csv_path /Volumes/storage/dataset/luft/google-landmark/train/train.csv \
    --cleaned_csv_path /Volumes/storage/dataset/luft/google-landmark/train/train_clean.csv
"""