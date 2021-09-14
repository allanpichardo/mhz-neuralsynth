import tensorflow as tf
from datetime import datetime

from datasets import SampleDataset


class SpectrogramCallback(tf.keras.callbacks.Callback):

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
            next_part = self.model.decoder(self.model.encoder(last_vector))
            wav_data = tf.concat([wav_data, next_part], axis=1)
            # wav_data = tf.concat([wav_data, initial_data], axis=1)
            last_vector = next_part
        return wav_data

    def on_epoch_end(self, epoch, logs=None):

        num_attempts = 3

        initial_data = []
        for x, y in self.train_dataset.shuffle(10240).take(1):
            for sample in x:
                if num_attempts > 0:
                    initial_data.append(sample)
                    num_attempts -= 1
                else:
                    break
            break

        synthesized = self._synthesize(initial_data, sample_length=self.sr)

        file_writer = tf.summary.create_file_writer(self.logdir)
        with file_writer.as_default():
            tf.summary.audio("Synthesis", synthesized, self.sr, step=epoch, max_outputs=num_attempts,
                             description="Synthesized audio")


if __name__ == '__main__':
    ds = SampleDataset().get_dataset()
    cb = SpectrogramCallback(ds)
    cb.on_epoch_end(1)
