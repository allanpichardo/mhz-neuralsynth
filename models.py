import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from losses import kl_divergence


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the STFT."""

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # feats = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Scalar(layers.Layer):

    def __init__(self, initial_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scalevar = tf.Variable(initial_value=initial_value, trainable=True)

    def call(self, inputs, *args, **kwargs):
        return tf.math.scalar_mul(self.scalevar, inputs)


class WaveGAN(keras.Model):

    def get_config(self):
        return super(WaveGAN, self).get_config()

    def __init__(self, waveform_shape=(16384, 1), shuffle_amount=2, batch_size=32, latent_dim=8, use_batch_norm=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n = shuffle_amount
        self._batch_size = batch_size
        self._latent_dim = latent_dim
        self._waveform_shape = waveform_shape
        self._use_batch_norm = use_batch_norm
        self._cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)
        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()

    def _get_conv_transpose_block(self, inputs, channels, strides=4):
        x = layers.Conv1DTranspose(channels, 25, strides=strides, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x) if self._use_batch_norm else x
        x = layers.LeakyReLU()(x)
        return x

    def _get_generator(self):
        inputs = layers.Input((self._latent_dim,))
        x = layers.Dense(4 * 4 * 1024, use_bias=False)(inputs)
        x = layers.BatchNormalization()(x) if self._use_batch_norm else x
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((4 * 4, 1024))(x)

        x = self._get_conv_transpose_block(x, 512)  # 64
        x = self._get_conv_transpose_block(x, 256)  # 256
        x = self._get_conv_transpose_block(x, 128)  # 1024
        x = self._get_conv_transpose_block(x, 64)  # 4096

        x = layers.Conv1DTranspose(1, 25, strides=4, padding='same', use_bias=False, activation='tanh')(x)  # 16384

        return keras.Model(inputs, x, name='WG_Generator')

    def _phase_shuffle(self, x, pad_type='reflect'):
        b, x_len, nch = x.get_shape().as_list()

        phase = tf.random.uniform([], minval=-self._n, maxval=self._n + 1, dtype=tf.int32)
        pad_l = tf.maximum(phase, 0)
        pad_r = tf.maximum(-phase, 0)
        phase_start = pad_r
        x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

        x = x[:, phase_start:phase_start + x_len]
        x.set_shape([b, x_len, nch])

        return x

    def _get_discriminator(self):
        inputs = layers.Input(self._waveform_shape)
        x = layers.Conv1D(64, 25, strides=4, padding='same')(inputs)
        x = layers.LeakyReLU()(x)
        x = self._phase_shuffle(x)

        x = layers.Conv1D(128, 25, strides=4, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = self._phase_shuffle(x)

        x = layers.Conv1D(256, 25, strides=4, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = self._phase_shuffle(x)

        x = layers.Conv1D(512, 25, strides=4, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = self._phase_shuffle(x)

        x = layers.Conv1D(1024, 25, strides=4, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = self._phase_shuffle(x)

        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)

        return keras.Model(inputs, x, name='WG_Discriminator')

    def _generator_loss(self, fake_output):
        return self._cross_entropy(tf.ones_like(fake_output), fake_output)

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self._cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def set_optimizers(self, generator_optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9),
                       discriminator_optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    def call(self, inputs, training=None, mask=None):
        return self.discriminator(inputs, training=training) if training else self.generator(inputs, training=training)

    def train_step(self, data):
        noise = tf.random.normal([self._batch_size, self._latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_waveforms = self.generator(noise, training=True)

            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_waveforms, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        }


class STFTInverter(keras.Model):

    def __init__(self, spectrogram_shape=(122, 129, 1), **kwargs):
        super(STFTInverter, self).__init__(**kwargs)
        self.spectrogram_shape = spectrogram_shape
        self.mcnn = self._get_mcnn()

    def _head(self, x):
        stride = 2
        kernel_width = 13
        i = 1
        channels = 2 ** (8 - 1)

        while i < 3:
            x = layers.Conv1DTranspose(channels, kernel_width, strides=stride)(x)
            x = layers.ELU()(x)
            i += 1

        x = layers.Conv1DTranspose(1, kernel_width, strides=stride)(x)
        x = layers.ELU()(x)

        x = Scalar()(x)
        return x

    def _scaled_softsign(self, x):
        x = tf.keras.activations.softsign(x)
        return Scalar()(x)

    def _get_mcnn(self):
        inputs = layers.Input(shape=self.spectrogram_shape)
        x = layers.Reshape((self.spectrogram_shape[0], self.spectrogram_shape[1]))(inputs)

        heads = []
        for i in range(8):
            heads.append(self._head(x))

        x = layers.Concatenate(axis=1)(heads)
        x = self._scaled_softsign(x)
        x = layers.Cropping1D((424, 0))(x)

        return tf.keras.Model(inputs, x)

    def call(self, inputs, training=None, mask=None):
        return self.mcnn(inputs)


class SpectrogramVAE(tf.keras.Model):

    def __init__(self, normalization_layer, n_fft=256, sample_length=8000, hop_size=64, window_length=256,
                 latent_dim=16, **kwargs):
        super(SpectrogramVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.n_fft = n_fft
        self.sample_length = sample_length
        self.hop_size = hop_size
        self.window_length = window_length
        self.normalization_layer = normalization_layer

        bins = int(n_fft // 2) + 1
        frames = int((sample_length - n_fft) // hop_size) + 1
        self.spectrogram_shape = (frames, bins, 1)

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _hybrid_pooling(self, inputs, pool_size=(2, 2), padding='valid'):
        return layers.Conv2D(int(inputs.shape[-1]), 1, padding='same')(
            layers.Concatenate()([
                layers.MaxPooling2D(pool_size, padding=padding)(inputs),
                layers.AveragePooling2D(pool_size, padding=padding)(inputs)
            ])
        )

    def _downsample_block(self, x, filters):
        x0 = x
        x = layers.Activation('elu')(x)
        x = layers.Conv2D(filters, 3, padding='same', data_format='channels_last')(x)
        x = layers.Activation('elu')(x)
        x = layers.Conv2D(filters, 1, padding='same')(x)
        x = layers.Activation('elu')(x)
        x = layers.Add()([x, x0])
        x = layers.AveragePooling2D()(x)
        return x

    def _upsample_blodk(self, x, filters):
        x0 = x
        x = layers.Activation('elu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same', data_format='channels_last')(x)
        x = layers.Activation('elu')(x)
        x = layers.Conv2DTranspose(filters, 1, padding='same')(x)
        x = layers.Activation('elu')(x)
        x = layers.Add()([x, x0])
        x = layers.UpSampling2D()(x)
        return x

    def _get_encoder(self):
        inputs = layers.Input(shape=self.spectrogram_shape)
        x = self.normalization_layer(inputs)

        x = layers.ZeroPadding2D((3, 0))(x)
        x = layers.Cropping2D(((0, 0), (1, 0)))(x)

        x = layers.Conv2D(64, 3, padding='same')(x)
        x = self._downsample_block(x, 64)
        x = self._downsample_block(x, 64)
        x = self._downsample_block(x, 64)
        # x = self._downsample_block(x, 128)

        z_mean = layers.Conv2D(self.latent_dim, 1, padding='same', activation='elu')(x)
        z_mean = layers.GlobalAveragePooling2D(name='z_mean')(z_mean)
        # z_mean = layers.Reshape((16, 16))(z_mean)

        z_log_var = layers.Conv2D(self.latent_dim, 1, padding='same', activation='elu')(x)
        z_log_var = layers.GlobalAveragePooling2D(name='z_log_var')(z_log_var)
        # z_log_var = layers.Conv2D(1, 1, padding='same', name="z_log_var")(z_log_var)
        # z_log_var = layers.Reshape((16, 16))(z_log_var)

        z = Sampling()([z_mean, z_log_var])

        return tf.keras.Model(inputs, [z_mean, z_log_var, z])

    def _get_decoder(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Reshape((1, 1, self.latent_dim))(inputs)
        x = layers.UpSampling2D(size=(16, 16))(x)
        x = layers.Conv2DTranspose(64, 1, padding='same', activation='elu')(x)

        x = self._upsample_blodk(x, 64)
        x = self._upsample_blodk(x, 64)
        x = self._upsample_blodk(x, 64)
        # x = self._upsample_blodk(x, 128)

        x = layers.Cropping2D((3, 0))(x)
        x = layers.ZeroPadding2D(((0, 0), (1, 0)))(x)

        x = layers.Conv2DTranspose(64, 3, padding='same')(x)
        out = layers.Conv2DTranspose(1, 1, padding='same')(x)

        return tf.keras.Model(inputs, out)

    def call(self, inputs, training=None, mask=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        self.encoder.add_loss(kl_divergence(z_mean, z_log_var))
        y_pred = self.decoder(z)
        return y_pred


class SampleVAE(tf.keras.Model):

    def __init__(self, vector_size=128, latent_dim=16, filters=32, **kwargs):
        super(SampleVAE, self).__init__(**kwargs)
        self.vector_size = vector_size
        self.latent_dim = latent_dim
        self.filters = filters

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _encoding_residual_block(self, x, filters=32, dilation_rate=1):
        x0 = x
        x = layers.ELU()(x)
        x = layers.Conv1D(filters, 3, dilation_rate=dilation_rate, padding='same')(x)
        x = layers.ELU()(x)
        x = layers.Conv1D(filters, 1, padding='same')(x)
        x = layers.Add()([x, x0])
        return x

    def _get_encoder(self):
        inputs = layers.Input(shape=(self.vector_size, 1))
        x = layers.Conv1D(self.filters, 3, padding='same')(inputs)
        for i in range(3):
            dilation = 1
            for j in range(10):
                x = self._encoding_residual_block(x, filters=self.filters, dilation_rate=dilation)
                dilation *= 2

        x = layers.Conv1D(self.latent_dim, 1, padding='same')(x)
        x = layers.AveragePooling1D(strides=self.latent_dim)(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def _wavenet_residual_block(self, x, filters=32, dilation_rate=1, bias=None, i=0):
        x0 = x

        if bias is not None:
            x = layers.Add(name="bias_{}".format(i))([x, bias])

        x = layers.Conv1D(filters, 3, dilation_rate=dilation_rate, padding='causal')(x)
        tan = layers.Activation('tanh')(x)
        sig = layers.Activation('sigmoid')(x)
        x = layers.Multiply()([tan, sig])
        skip = layers.Conv1D(filters, 1, padding='same')(x)
        resid = layers.Add()([skip, x0])
        return resid, skip

    def _get_decoder(self):
        signal_input = layers.Input(shape=(self.vector_size, 1))
        latent_input = layers.Input(shape=(self.vector_size // self.latent_dim, self.latent_dim))

        x = layers.Conv1D(self.filters, 3, padding='causal')(signal_input)

        z = layers.UpSampling1D(self.latent_dim)(latent_input)
        z_dimensions = tf.split(z, self.latent_dim, axis=-1)

        dilation = 1
        skip_connections = []
        for i in range(self.latent_dim):
            z_bias = z_dimensions[i]
            z_bias = layers.Conv1D(self.filters, 3, padding='causal')(z_bias)

            x, skip = self._wavenet_residual_block(x, filters=self.filters, dilation_rate=dilation, bias=z_bias, i=i)
            skip_connections.append(skip)
            dilation *= 2

        x = layers.Add()(skip_connections)
        x = layers.ELU()(x)
        x = layers.Conv1D(self.filters, 1, padding='same', strides=2)(x)
        x = layers.ELU()(x)
        x = layers.Conv1D(self.filters, 1, padding='same', strides=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='softmax')(x)

        return tf.keras.Model(inputs=[signal_input, latent_input], outputs=x)

    def call(self, inputs, training=None, mask=None):
        z = self.encoder(inputs)
        y_pred = self.decoder([inputs, z])
        return y_pred


def get_test_data(batch_size=1):
    n = tf.random.normal((122, 129, 1))
    return tf.expand_dims(n, 0)


if __name__ == '__main__':
    gan = WaveGAN()
    generator = gan._get_generator()
    generator.summary()

    z = tf.random.normal([1, 8])
    wav = generator([z], training=False)

    discriminator = gan._get_discriminator()
    discriminator.summary()

    decision = discriminator(wav)
    print(decision)
