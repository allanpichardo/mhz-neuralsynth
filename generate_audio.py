import tensorflow as tf
import os
from models import SpectrogramVAE
import librosa
import numpy as np


def rescale(x):
    min = np.min(x)
    max = np.max(x)
    scaled = (x - min) / (max - min)
    # scaled *= 255.0
    return scaled


def lerp(a, b, t):
    return a + (b - a) * t


def main():
    latent_dim = 8
    version = '2'
    generator_model_path = os.path.join(os.path.dirname(__file__), 'models', 'generator_mod_v{}'.format(version))
    disc_model_path = os.path.join(os.path.dirname(__file__), 'models', 'discriminator_mod_v{}'.format(version))

    generator = None
    if os.path.exists(generator_model_path) and os.path.exists(disc_model_path):
        print("Found saved model, loading weights")
        generator = tf.keras.models.load_model(generator_model_path, compile=False)
    else:
        print("Couldn't find model")
        exit(1)

    outdir = os.path.join(os.path.dirname(__file__), 'generated', 'wavegan', version)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    amount = 20
    print("Generating {} samples...".format(amount))
    a = tf.squeeze(tf.random.normal([1, latent_dim], stddev=1.0)).numpy()
    b = tf.squeeze(tf.random.normal([1, latent_dim], stddev=1.0)).numpy()
    vecs = []
    for i in range(amount):
        v = np.squeeze(lerp(a, b, i / amount))
        vecs.append(v)
    vecs = tf.convert_to_tensor(vecs)
    outputs = generator(vecs)
    print("Done.")
    print("Writing to disk...")

    i = 1
    for wav in outputs:
        encoded = tf.audio.encode_wav(wav, 16000)
        tf.io.write_file(os.path.join(outdir, "generated-{}.wav".format(i)), encoded)
        i += 1
        print(".", end="\r", flush=True)

    print("Done.")


if __name__ == '__main__':
    main()
