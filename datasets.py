import tensorflow as tf
import tensorflow.keras as keras
import librosa
import os
import glob
import numpy as np
from utils import mu_law_encode
from sklearn.model_selection import train_test_split
import kapre
import json


class SpectrogramDataset:

    def __init__(self, sample_rate=16000, n_fft=256, sample_length=8000, hop_size=64, window_length=256, subset='train', data_sample_length=8000, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.sample_length = sample_length
        self.hop_size = hop_size
        self.window_length = window_length
        self.data_sample_length = data_sample_length
        self.n_mels = n_mels

        dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'train*' if subset == 'train' else 'validation*')

        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(
            glob.glob(dataset_path),
            compression_type='ZLIB'
        )  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order

        self.dataset = dataset.map(
            self._read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            self._wav_normalize, num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            self._to_spectrogram, num_parallel_calls=tf.data.AUTOTUNE
        ).cache()

    def _wav_normalize(self, example):
        wav = example['x']
        mean = tf.math.reduce_mean(wav, axis=0)
        std = tf.math.reduce_std(wav, axis=0)
        return (wav - mean) / std

    def _to_spectrogram(self, x):
        data = x

        m = tf.expand_dims(data, 0)
        m = kapre.STFT(n_fft=self.n_fft, win_length=self.window_length, hop_length=self.hop_size, input_data_format='channels_last', output_data_format='channels_last')(m)
        m = kapre.Magnitude()(m)
        m = kapre.MagnitudeToDecibel()(m)
        m = tf.squeeze(m, axis=0)
        return m, m

    def _read_tfrecord(self, raw):
        feature_description = {
            'x': tf.io.FixedLenFeature([self.data_sample_length, 1], tf.float32,
                                       default_value=np.zeros((self.data_sample_length,))),
        }

        example = tf.io.parse_single_example(raw, feature_description)
        return example

    def get_dataset(self, batch_size=16, shuffle_buffer=102400):
        return self.dataset.shuffle(shuffle_buffer).prefetch(tf.data.AUTOTUNE).batch(batch_size)

    def get_normalization_layer(self):
        norm_layer = keras.layers.Normalization(name='normalization')
        print("Adapting normalization layer...")
        dataset = self.get_dataset(shuffle_buffer=1).map(lambda x, y: x)
        norm_layer.adapt(dataset)
        print("Done.")
        return norm_layer


class SampleDataset:

    def __init__(self, vector_size=128, data_sample_length=8000, subset='train', full_set=True, stride=128):
        self.vector_size = vector_size
        self.data_sample_length = data_sample_length
        self.stride = stride

        dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'train*' if subset == 'train' else 'validation*')

        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(
            glob.glob(dataset_path),
            compression_type='ZLIB'
        )  # automatically interleaves reads from multiple files
        if not full_set:
            dataset = tf.data.TFRecordDataset(
                glob.glob(dataset_path)[0],
                compression_type='ZLIB'
            )

        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
        self.dataset = dataset.map(
            self._read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            self._split_into_chunks, num_parallel_calls=tf.data.AUTOTUNE
        ).flat_map(
            self._flatten
        ).map(
            self._quantize, num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            self._y_to_one_hot_byte, num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            self._normalize, num_parallel_calls=tf.data.AUTOTUNE
        ).cache()

    def _y_to_one_hot_byte(self, x, y):
        layer = tf.keras.layers.CategoryEncoding(num_tokens=256, output_mode="one_hot")
        onehot = layer(y)
        return x, onehot

    def _split_into_chunks(self, dataset):
        wav = dataset['x']

        start = 0
        max_len = self.data_sample_length - (self.vector_size + 1)

        X = tf.reshape(tf.convert_to_tensor(()), [0, self.vector_size, 1])
        Y = tf.reshape(tf.convert_to_tensor(()), [0, 1])

        while start < max_len:
            end = start + self.vector_size + 1
            part = wav[start:end]
            y = part[-1]
            y = tf.expand_dims(y, axis=0)

            x = part[:-1]
            x = tf.expand_dims(x, axis=0)

            X = tf.concat([X, x], 0)
            Y = tf.concat([Y, y], 0)
            start += self.stride

        return X, Y

    def _flatten(self, x, y):
        return tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(x),
            tf.data.Dataset.from_tensor_slices(y)
        ))

    def _quantize(self, x, y):
        return mu_law_encode(x), mu_law_encode(y)

    def _normalize(self, x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    def _read_tfrecord(self, raw):
        feature_description = {
            'x': tf.io.FixedLenFeature([self.data_sample_length, 1], tf.float32,
                                       default_value=np.zeros((self.data_sample_length,))),
        }

        example = tf.io.parse_single_example(raw, feature_description)
        return example

    def get_dataset(self, batch_size=16, shuffle_buffer=102400):
        return self.dataset.shuffle(shuffle_buffer).prefetch(tf.data.AUTOTUNE).batch(batch_size)


class SampleDatasetBuilder:

    def __init__(self, sample_length=8000, sr=16000,
                 sample_dir=os.path.join(os.path.dirname(__file__), 'samples')) -> None:
        super().__init__()
        self.sample_dir = sample_dir
        self.sample_length = sample_length
        self.sr = sr
        self.wav_paths = glob.glob(os.path.join(sample_dir, '*', '*'))
        self.train_paths, self.test_paths = train_test_split(self.wav_paths, test_size=0.20, shuffle=True)

    @staticmethod
    def get_tfrecord_options():
        return tf.io.TFRecordOptions(
            compression_type='ZLIB'
        )

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def serialize_example(self, x):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {
            'x': SampleDatasetBuilder._float_feature(x),
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def save_record_file(self, subset='train'):
        filepath = os.path.join(os.path.dirname(__file__), 'data',
                                'train.tfrecord' if subset != 'validation' else 'validation.tfrecord')

        if not os.path.exists(os.path.dirname(filepath)):
            print("Path not found, creating direcory")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        print("Creating file writer for {}".format(filepath))
        with tf.io.TFRecordWriter(filepath, options=SampleDatasetBuilder.get_tfrecord_options()) as writer:
            for i, path in enumerate(self.train_paths if subset != 'validation' else self.test_paths):
                print("Opening wav file {}".format(path))
                wav, sr = librosa.load(path, sr=self.sr, res_type="kaiser_fast")
                wav, index = librosa.effects.trim(wav)
                # f0, voiced_flag, voiced_probs = librosa.pyin(wav, fmin=librosa.note_to_hz('C1'),
                #                                              fmax=librosa.note_to_hz('C9'))
                # pitch = np.average(f0[~np.isnan(f0)])

                start = 0
                while start < wav.size - self.sample_length:
                    x = wav[start:start + self.sample_length]
                    x = np.expand_dims(x, 1)
                    example = self.serialize_example(x)
                    writer.write(example)

                    start += self.sample_length
                    print(".", end="")
                print("\n")
            print("Done. Closing file")

    @staticmethod
    def shard_record(filepath, num_shards):
        dir = os.path.dirname(filepath)
        name = os.path.basename(filepath)

        print("Loading dataset from {}".format(filepath))
        raw_dataset = tf.data.TFRecordDataset([filepath], compression_type='ZLIB')
        for i in range(num_shards):
            print("Writing part {}".format(i))
            with tf.io.TFRecordWriter(os.path.join(dir, "{}-part-{}.tfrecord".format(name, i)),
                                      SampleDatasetBuilder.get_tfrecord_options()) as writer:
                shard = raw_dataset.shard(num_shards, i)
                for example in shard:
                    writer.write(example.numpy())
        print("Done.")


if __name__ == '__main__':
    tf.data.experimental.enable_debug_mode()

    st = SpectrogramDataset()
    # norm_layer = st.get_normalization_layer()

    ds = st.get_dataset(shuffle_buffer=1)
    for X, Y in ds:
        # print(norm_layer(X))
        print(X)
        exit(0)

    # builder = SampleDatasetBuilder()
    # builder.save_record_file(subset='train')
    # builder.save_record_file(subset='validation')
    # SampleDatasetBuilder.shard_record('data/train.tfrecord', 14)
    # SampleDatasetBuilder.shard_record('data/validation.tfrecord', 4)
