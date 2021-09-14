import tensorflow as tf
from datetime import datetime
import io
from matplotlib import pyplot as plt
from datasets import SampleDataset


class SynthesisCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_dataset, vector_size=128, sr=16000, logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")):
        super().__init__()
        self.logdir = logdir
        self.sr = sr
        self.vector_size = vector_size
        self.train_dataset = train_dataset

    def _synthesize(self, initial_data, sample_length=16000):
        passes = sample_length // self.vector_size
        wav_data = tf.identity(initial_data)
        last_vector = wav_data
        for i in range(passes):
            u, v, z = self.model.encoder(last_vector)
            next_part = self.model.decoder(z)
            wav_data = tf.concat([wav_data, next_part], axis=1)
            # wav_data = tf.concat([wav_data, initial_data], axis=1)
            last_vector = next_part
        return wav_data

    def _gen_plot(self, wave):
        """Create a pyplot plot and save to buffer."""
        w = tf.squeeze(wave, axis=-1)
        plt.figure()
        plt.plot(w[0].numpy())
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
        img = tf.expand_dims(img, axis=0)
        plt.close()
        return img

    def on_epoch_end(self, epoch, logs=None):

        num_attempts = 3

        initial_data = []
        for x, y in self.train_dataset.shuffle(1024).take(1):
            for sample in x:
                if num_attempts > 0:
                    initial_data.append(sample)
                    num_attempts -= 1
                else:
                    break
            break

        synthesized = self._synthesize(initial_data, sample_length=self.sr)

        plots = []
        for waveform in synthesized:
            plots.append(self._gen_plot(waveform))

        file_writer = tf.summary.create_file_writer(self.logdir)
        with file_writer.as_default():
            tf.summary.audio("Synthesis", synthesized, self.sr, step=epoch, max_outputs=num_attempts,
                             description="Synthesized audio")
            tf.summary.image("Synthesized Waveform", plots, step=epoch, max_outputs=num_attempts)


if __name__ == '__main__':
    ds = SampleDataset().get_dataset()
    cb = SynthesisCallback(ds)
    cb.on_epoch_end(1)
