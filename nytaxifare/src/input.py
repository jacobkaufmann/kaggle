from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os

import tensorflow as tf
import pandas as pd
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "", "Data directory")
tf.app.flags.DEFINE_string("train_file", "", "File containing training data")
tf.app.flags.DEFINE_string("test_file", "", "File containing test data")

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def preprocess_data(path):
    print("Reading", path)
    data = pd.read_csv(path)
    date = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S %Z")
            for x in data["pickup_datetime"]]
    data["pickup_datetime_month"] = [x.month for x in date]
    data["pickup_datetime_day"] = [x.weekday() for x in date]
    data["pickup_datetime_hour"] = [x.hour for x in date]
    data.drop(columns=["key", "pickup_datetime"], inplace=True)
    return data.values

def train_eval_split(data, train_pct=.95):
    # Split training data into train and validation sets
    np.random.shuffle(data)
    split = int(len(data) * train_pct)
    return data[:split], data[split:]

def compose_tfrecord(data, path, dataset):
    # Write dataset to tfrecord file
    print("Writing", path)
    with tf.python_io.TFRecordWriter(path=path) as writer:
        for row in data:
            feature_dict = {}
            if dataset == "train" or dataset == "eval":
                feature_dict = {
                    "label": _float_feature(row[0]),
                    "pickup_longitude": _float_feature(row[1]),
                    "pickup_latitude": _float_feature(row[2]),
                    "dropoff_longitude": _float_feature(row[3]),
                    "dropoff_latitude": _float_feature(row[4]),
                    "passenger_count": _int64_feature(int(row[5])),
                    "pickup_datetime_month": _int64_feature(int(row[6])),
                    "pickup_datetime_day": _int64_feature(int(row[7])),
                    "pickup_datetime_hour": _int64_feature(int(row[8]))
                }
            elif dataset == "test":
                feature_dict = {
                    "pickup_longitude": _float_feature(row[0]),
                    "pickup_latitude": _float_feature(row[1]),
                    "dropoff_longitude": _float_feature(row[2]),
                    "dropoff_latitude": _float_feature(row[3]),
                    "passenger_count": _int64_feature(int(row[4])),
                    "pickup_datetime_month": _int64_feature(int(row[5])),
                    "pickup_datetime_day": _int64_feature(int(row[6])),
                    "pickup_datetime_hour": _int64_feature(int(row[7]))
                }
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature_dict
                )
            )
            writer.write(example.SerializeToString())
    print("Finished writing to", path)

def main(unused_arg):
    if os.path.isdir(FLAGS.data_dir):
        train_path = os.path.join(FLAGS.data_dir, FLAGS.train_file)
        if os.path.isfile(train_path):
            train_data = preprocess_data(train_path)
            train, val = train_eval_split(train_data)
            compose_tfrecord(train, os.path.join(FLAGS.data_dir, "train.tfrecord"),
                             dataset="train")
            compose_tfrecord(val, os.path.join(FLAGS.data_dir, "eval.tfrecord"),
                             dataset="eval")

        test_path = os.path.join(FLAGS.data_dir, FLAGS.test_file)
        if os.path.isfile(test_path):
            test = preprocess_data(test_path)
            compose_tfrecord(test, os.path.join(FLAGS.data_dir, "test.tfrecord"),   
                             dataset="test")
    else:
        print("Invalid path")

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
