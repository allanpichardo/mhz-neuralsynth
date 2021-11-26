import os
import time

import tensorflow as tf

from datasets import SampleDataset
from models import WaveGAN


def generate_and_save_audio(generator, epoch, test_input, sample_rate=16000):
    predictions = generator(test_input, training=False)

    for i in range(predictions.shape[0]):
        outdir = os.path.join(os.path.dirname(__file__), 'generated')
        audiofile = os.path.join(outdir, "generated_epoch_{}__{}.wav".format(epoch, i))
        wav = tf.audio.encode_wav(predictions[i, :, :], sample_rate)
        tf.io.write_file(audiofile, wav)
        print("{}%".format((i * 100) / predictions.shape[0]), end="\r", flush=True)

    print("Samples generated.")


def main():
    version = '1'
    sr = 16000
    batch_size = 32
    latent_dim = 8
    epochs = 2000
    learning_rate = 0.0001
    num_examples_to_generate = 4
    seed = tf.random.normal([num_examples_to_generate, latent_dim])

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

    generator_model_path = os.path.join(os.path.dirname(__file__), 'models', 'generator_mod_v{}'.format(version))
    discriminator_model_path = os.path.join(os.path.dirname(__file__), 'models', 'discriminator_mod_v{}'.format(version))

    dataset = SampleDataset()

    wavegan = WaveGAN(batch_size=batch_size, latent_dim=latent_dim)

    if os.path.exists(generator_model_path) and os.path.exists(discriminator_model_path):
        print("Found saved model, loading weights")
        wavegan.generator = tf.keras.models.load_model(generator_model_path, compile=False)
        wavegan.discriminator = tf.keras.models.load_model(discriminator_model_path, compile=False)

    wavegan.generator.summary()
    wavegan.discriminator.summary()

    train_dataset = dataset.get_dataset(batch_size=batch_size, shuffle_buffer=102400)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=wavegan.generator_optimizer,
                                     discriminator_optimizer=wavegan.discriminator_optimizer,
                                     generator=wavegan.generator,
                                     discriminator=wavegan.discriminator)

    # training loop here
    print("Starting training...")
    print("")

    for epoch in range(epochs):
        print("Starting epoch {} of {}:".format(epoch, epochs))

        start = time.time()

        i = 1
        max = train_dataset.cardinality().numpy()
        for wav_batch in train_dataset:
            wavegan.train_step(wav_batch)
            print("Progress: {}%".format((i * 100) / max))

        generate_and_save_audio(wavegan.generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    generate_and_save_audio(wavegan.generator, epochs, seed)

    wavegan.generator.save(generator_model_path, save_format='tf', include_optimizer=False)
    wavegan.discriminator.save(discriminator_model_path, save_format='tf', include_optimizer=False)


if __name__=='__main__':
    main()
