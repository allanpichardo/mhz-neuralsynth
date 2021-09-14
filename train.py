import tensorflow as tf
import os
from datetime import datetime

from callbacks import SynthesisCallback
from datasets import SampleDataset
from models import SampleVAE


def main():
    version = 1
    sr = 16000
    batch_size = 16
    vector_size = 128
    latent_dim = 16
    epochs = 200
    learning_rate = 0.001

    logdir = os.path.join(os.path.dirname(__file__), 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

    enc_model_path = os.path.join(os.path.dirname(__file__), 'models', 'enc_mod_v{}'.format(version))
    dec_model_path = os.path.join(os.path.dirname(__file__), 'models', 'dec_mod_v{}'.format(version))

    autoencoder = SampleVAE(vector_size=vector_size,latent_dim=latent_dim)
    if os.path.exists(enc_model_path) and os.path.exists(dec_model_path):
        print("Found saved model, loading weights")
        autoencoder.encoder = tf.keras.models.load_model(enc_model_path, compile=False)
        autoencoder.decoder = tf.keras.models.load_model(dec_model_path, compile=False)

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    dataset = SampleDataset(vector_size=vector_size).get_dataset(batch_size=batch_size)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    autoencoder.fit(dataset, epochs=epochs, callbacks=[
        SynthesisCallback(dataset, vector_size=vector_size, sr=sr, logdir=logdir),
        tf.keras.callbacks.TensorBoard(log_dir=logdir, embeddings_freq=1)
    ])

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

    autoencoder.encoder.save(enc_model_path, save_format='tf', include_optimizer=False)
    autoencoder.decoder.save(dec_model_path, save_format='tf', include_optimizer=False)


if __name__=='__main__':
    main()