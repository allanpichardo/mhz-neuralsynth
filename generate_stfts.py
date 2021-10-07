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


def main():
    version = 'stft_n2_3'
    enc_model_path = os.path.join(os.path.dirname(__file__), 'models', 'enc_mod_v{}'.format(version))
    dec_model_path = os.path.join(os.path.dirname(__file__), 'models', 'dec_mod_v{}'.format(version))

    decoder = None
    if os.path.exists(enc_model_path) and os.path.exists(dec_model_path):
        print("Found saved model, loading weights")
        decoder = tf.keras.models.load_model(dec_model_path, compile=False)
    else:
        print("Couldn't find model")
        exit(1)

    outdir = os.path.join(os.path.dirname(__file__), 'generated', 'stft', version)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print("Generating spectrograms...")
    vecs = np.random.uniform(-0.5, 100.0, (20, 16, 16))
    outputs = decoder(vecs)
    print("Done.")

    print("Saving Images. ", end="")
    i = 1
    for spec in outputs:
        imgfile = os.path.join(outdir, "{}.jpg".format(i))
        audiofile = os.path.join(outdir, "{}.wav".format(i))
        data = rescale(spec)
        tf.keras.utils.save_img(imgfile, data, data_format='channels_last')
        S = tf.squeeze(spec, 2).numpy()
        S = librosa.db_to_power(S)
        wav = librosa.griffinlim(S, n_iter=128, hop_length=64)
        wav = np.expand_dims(wav, 1)
        wav = tf.audio.encode_wav(wav, 16000)
        tf.io.write_file(audiofile, wav)
        print("{}%".format((i * 100)/20), end="\r", flush=True)
        i += 1

    print("Generating audio via GL")


    print("Finished.")


if __name__ == '__main__':
    main()
