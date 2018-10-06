from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "", "Data directory")
tf.app.flags.DEFINE_string("filename", "", "Filename for train file")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.app.flags.DEFINE_string("mode", "", "train")
tf.app.flags.DEFINE_string("model_dir", "./models", "Model directory")

TRAIN_PATH = os.path.join(FLAGS.data_dir, FLAGS.filename)

def train_input_fn():
    if os.path.isfile(TRAIN_PATH):
        dataset = tf.data.TFRecordDataset(TRAIN_PATH)

        def _parser(record):
            keys_to_features = {
                "pickup_longitude": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                "pickup_latitude": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                "dropoff_longitude": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                "dropoff_latitude": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                "passenger_count": tf.FixedLenFeature((), tf.int64, default_value=1),
                "pickup_datetime_month": tf.FixedLenFeature((), tf.int64, default_value=1),
                "pickup_datetime_day": tf.FixedLenFeature((), tf.int64, default_value=1),
                "pickup_datetime_hour": tf.FixedLenFeature((), tf.int64, default_value=12),
                "label": tf.FixedLenFeature((), tf.float32, default_value=0.0)
            }
            parsed = tf.parse_single_example(record, keys_to_features)
            print(parsed)
            return {
                "pickup_longitude": parsed["pickup_longitude"],
                "pickup_latitude": parsed["pickup_latitude"],
                "dropoff_longitude": parsed["dropoff_longitude"],
                "dropoff_latitude": parsed["dropoff_latitude"],
                "passenger_count": parsed["passenger_count"],
                "pickup_datetime_month": parsed["pickup_datetime_month"],
                "pickup_datetime_day": parsed["pickup_datetime_day"],
                "pickup_datetime_hour": parsed["pickup_datetime_hour"]
            }, parsed["label"]

        print(dataset)
        dataset = dataset.map(_parser)
        dataset = dataset.shuffle(buffer_size=1000000).repeat().batch(FLAGS.batch_size)
        print(dataset)
        return dataset
    else:
        print("Invalid path")

def feature_cols():
    # Create feature columns for Estimator
    longitude_boundaries = [-74.5, -74.2, -74.0, -73.8,
                            -73.6, -73.5, -73.4, -73.2, -73.0, -72.8, -72.5]
    latitude_boundaries = [40.0, 40.3, 40.5, 40.7, 40.9,
                           41.1, 41.3, 41.5, 41.7, 41.9, 42.2]

    pickup_date_month = tf.feature_column.numeric_column(
        key="pickup_datetime_month", dtype=tf.int32
    )
    pickup_date_day = tf.feature_column.numeric_column(
        key="pickup_datetime_day", dtype=tf.int32
    )
    pickup_date_hour = tf.feature_column.numeric_column(
        key="pickup_datetime_hour", dtype=tf.int32
    )
    pickup_longitude = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column(
            key="pickup_longitude", dtype=tf.float32
        ),
        boundaries=longitude_boundaries
    )
    pickup_latitude = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column(
            key="pickup_latitude", dtype=tf.float32
        ),
        boundaries=latitude_boundaries,
    )
    crossed_pickup = tf.feature_column.crossed_column(
        [pickup_longitude, pickup_latitude], 10000
    )
    dropoff_longitude = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column(
            key="dropoff_longitude", dtype=tf.float32
        ),
        boundaries=longitude_boundaries
    )
    dropoff_latitude = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column(
            key="dropoff_latitude", dtype=tf.float32
        ),
        boundaries=latitude_boundaries,
    )
    crossed_dropoff = tf.feature_column.crossed_column(
        [dropoff_longitude, dropoff_latitude], 10000
    )
    passenger_count = tf.feature_column.numeric_column(
        key="passenger_count", dtype=tf.int32
    )

    return [
        pickup_longitude,
        pickup_latitude,
        tf.feature_column.embedding_column(crossed_pickup, 10),
        dropoff_longitude,
        dropoff_latitude,
        tf.feature_column.embedding_column(crossed_dropoff, 10),
        passenger_count,
        pickup_date_month,
        pickup_date_day,
        pickup_date_hour,
    ]

def train(est, steps=50000):
    est.train(steps=steps, input_fn=train_input_fn)
    
def evaluate():
    pass

def main(unused_arg):
    # Initialize Estimator and begin training
    est = tf.estimator.DNNRegressor(
        hidden_units=[1024, 1024, 512, 256, 128, 64, 32, 16],
        feature_columns=feature_cols(),
        model_dir="../models",
        dropout=.3
    )

    if FLAGS.mode == "train":
        train(est)
    elif FLAGS.mode == "eval":
        pass

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
