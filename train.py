import tensorflow as tf
import os
from datetime import datetime

from callbacks import SynthesisCallback
from datasets import SampleDataset
from models import SampleVAE


def main():
    version = 1
    sr = 16000
    batch_size = 64
    vector_size = 128
    stride = vector_size // 2 #make this smaller
    filters = 32
    latent_dim = 8
    epochs = 2000
    learning_rate = 0.00001

    logdir = os.path.join(os.path.dirname(__file__), 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

    enc_model_path = os.path.join(os.path.dirname(__file__), 'models', 'enc_mod_v{}'.format(version))
    dec_model_path = os.path.join(os.path.dirname(__file__), 'models', 'dec_mod_v{}'.format(version))

    autoencoder = SampleVAE(vector_size=vector_size, latent_dim=latent_dim, filters=filters)
    if os.path.exists(enc_model_path) and os.path.exists(dec_model_path):
        print("Found saved model, loading weights")
        autoencoder.encoder = tf.keras.models.load_model(enc_model_path, compile=False)
        autoencoder.decoder = tf.keras.models.load_model(dec_model_path, compile=False)

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    tran_dataset = SampleDataset(vector_size=vector_size, subset='train', stride=stride).get_dataset(batch_size=batch_size)
    val_dataset = SampleDataset(vector_size=vector_size, subset='validation', stride=stride).get_dataset(batch_size=batch_size)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        metrics=['accuracy'],
                        loss='categorical_crossentropy')

    autoencoder.fit(tran_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[
        # SynthesisCallback(tran_dataset, vector_size=vector_size, sr=sr, logdir=logdir),
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    ])

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

    autoencoder.encoder.save(enc_model_path, save_format='tf', include_optimizer=False)
    autoencoder.decoder.save(dec_model_path, save_format='tf', include_optimizer=False)


if __name__=='__main__':
    main()
