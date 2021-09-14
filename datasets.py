import tensorflow as tf
import librosa
import os
import glob
import numpy as np


class SampleDataset:

    def __init__(self, vector_size=128, dataset_path=os.path.join(os.path.dirname(__file__), 'data', '*.tfrecord')):
        self.vector_size = vector_size

        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(
            glob.glob(dataset_path)
        )  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
        self.dataset = dataset.map(
            self._read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
        )

    def _read_tfrecord(self, raw):
        feature_description = {
            'x': tf.io.FixedLenFeature([self.vector_size], tf.float32, default_value=np.zeros((self.vector_size,))),
            'y': tf.io.FixedLenFeature([self.vector_size], tf.float32, default_value=np.zeros((self.vector_size,))),
        }

        example = tf.io.parse_single_example(raw, feature_description)
        return example["x"], example["y"]

    def get_dataset(self, batch_size=16):
        return self.dataset.shuffle(10240).prefetch(tf.data.AUTOTUNE).batch(batch_size)


class SampleDatasetBuilder:

    def __init__(self, vector_size=128, sr=16000,
                 sample_dir=os.path.join(os.path.dirname(__file__), 'samples')) -> None:
        super().__init__()
        self.sample_dir = sample_dir
        self.vector_size = vector_size
        self.sr = sr
        self.wav_paths = glob.glob(os.path.join(sample_dir, '*', '*'))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def serialize_example(self, x, y):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {
            'x': SampleDatasetBuilder._float_feature(x),
            'y': SampleDatasetBuilder._float_feature(y),
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def save_record_file(self, filepath=os.path.join(os.path.dirname(__file__), 'data', 'samples.tfrecord')):
        if not os.path.exists(os.path.dirname(filepath)):
            print("Path not found, creating direcory")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        print("Creating file writer for {}".format(filepath))
        with tf.io.TFRecordWriter(filepath) as writer:
            for i, path in enumerate(self.wav_paths):
                print("Opening wav file {}".format(path))
                wav, sr = librosa.load(path, sr=self.sr, res_type="kaiser_fast")
                start = self.vector_size
                while start < wav.size - self.vector_size:
                    x_start = start - self.vector_size

                    y = wav[start:start + self.vector_size]
                    x = wav[x_start:x_start + self.vector_size]

                    example = self.serialize_example(x, y)
                    writer.write(example)

                    start += self.vector_size
                    print(".")
            print("Done. Closing file")

    @staticmethod
    def shard_record(filepath, num_shards):
        dir = os.path.dirname(filepath)
        name = os.path.basename(filepath)

        print("Loading dataset from {}".format(filepath))
        raw_dataset = tf.data.TFRecordDataset([filepath])
        for i in range(num_shards):
            print("Writing part {}".format(i))
            with tf.io.TFRecordWriter(os.path.join(dir, "{}-part-{}.tfrecord".format(name, i))) as writer:
                shard = raw_dataset.shard(num_shards, i)
                for example in shard:
                    writer.write(example.numpy())
        print("Done.")


if __name__ == '__main__':
    ds = SampleDataset()
    dataset = ds.get_dataset()

    for x, y in dataset.take(1):
        for ex in x:
            print(ex)
