import keras
import tensorflow as tf
from datetime import datetime
import io
from matplotlib import pyplot as plt
from datasets import SampleDataset, SpectrogramDataset
from models import SampleVAE, SpectrogramVAE
from utils import mu_law_decode
from griffin_lim import TFGriffinLim


class SpectrogramCallback(tf.keras.callbacks.Callback):

    def __init__(self, dataset, sr=16000, logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")):
        super(SpectrogramCallback, self).__init__()
        self.dataset = dataset
        self.logdir = logdir
        self.sr = sr

    def on_epoch_end(self, epoch, logs=None):
        self.model = SpectrogramVAE(keras.layers.Normalization())
        norm = self.model.encoder.get_layer('normalization')

        batch = self.dataset
        batch = batch.map(lambda x, y: x).take(1)
        test_batch = []
        for x in batch:
            test_batch = x
            break

        in_normed = norm(test_batch)
        in_normed = self._rescale(in_normed)

        out_normed = self.model(test_batch)
        out_normed = self._rescale(out_normed)

        file_writer = tf.summary.create_file_writer(self.logdir)
        with file_writer.as_default():
            tf.summary.image("Input Spectrogram", in_normed, step=epoch, max_outputs=4)
            tf.summary.image("Output Spectrogram", out_normed, step=epoch, max_outputs=4)

    def _rescale(self, x):
        min = tf.math.reduce_min(x)
        max = tf.math.reduce_max(x)
        return (x - min) / (max - min)


class SynthesisCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_dataset, vector_size=128, sr=16000,
                 logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")):
        super().__init__()
        self.logdir = logdir
        self.sr = sr
        self.vector_size = vector_size
        self.train_dataset = train_dataset

    def _reconstruct(self, one_hot):
        a = tf.argmax(one_hot, axis=1)
        a = tf.cast(a, tf.float32)
        a = tf.expand_dims(a, 1)
        a = tf.expand_dims(a, 2)
        return a / 255.0

    def _synthesize(self, initial_data, sample_length=16000):
        model = SampleVAE()
        passes = sample_length - self.vector_size
        wav_data = tf.identity(initial_data)
        for i in range(passes):
            signal_input = wav_data[:, -self.vector_size:, :]
            print("Progress: {}%".format(float(i) / float(passes) * 100.0), end="\r", flush=True)

            z = model.encoder(signal_input)
            # z = self.model.encoder(signal_input)

            next_sample = model.decoder([signal_input, z])
            # next_sample = self.model.decoder([signal_input, z])

            next_sample = self._reconstruct(next_sample)
            wav_data = tf.concat([wav_data, next_sample], axis=1)

        wav_data = wav_data * 255.0
        return mu_law_decode(wav_data)

    def _gen_plot(self, w):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.plot(w[0].numpy())
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
        plt.close()
        return img

    def on_epoch_end(self, epoch, logs=None):

        num_attempts = 3

        initial_data = []
        for x, y in self.train_dataset.shuffle(64).take(1):
            for sample in x:
                if num_attempts > 0:
                    initial_data.append(sample)
                    num_attempts -= 1
                else:
                    break
            break

        synthesized = self._synthesize(initial_data, sample_length=self.sr)

        tf.io.write_file("epoch-{}.wav".format(epoch), tf.audio.encode_wav(synthesized[0], self.sr))

        plots = []
        for waveform in synthesized:
            plots.append(self._gen_plot(waveform))

        file_writer = tf.summary.create_file_writer(self.logdir)
        with file_writer.as_default():
            tf.summary.audio("Synthesis", synthesized, self.sr, step=epoch, max_outputs=num_attempts,
                             description="Synthesized audio")
            tf.summary.image("Synthesized Waveform", plots, step=epoch, max_outputs=num_attempts)


if __name__ == '__main__':
    tf.data.experimental.enable_debug_mode()
    tf.config.run_functions_eagerly(True)
    ds = SpectrogramDataset().get_dataset(batch_size=16, shuffle_buffer=1).take(1)
    cb = SpectrogramCallback(ds)
    cb.on_epoch_end(1)
