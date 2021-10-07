import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers

from datasets import SpectrogramDataset
from losses import kl_divergence


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the STFT."""

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        feats = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim, feats))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Scalar(layers.Layer):

    def __init__(self, initial_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scalevar = tf.Variable(initial_value=initial_value, trainable=True)

    def call(self, inputs, *args, **kwargs):
        return tf.math.scalar_mul(self.scalevar, inputs)



class STFTInverter(tensorflow.keras.Model):

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
        x = layers.Conv2D(filters, 7, padding='same', data_format='channels_last')(x)
        x = layers.Activation('elu')(x)
        x = layers.Conv2D(filters, 1, padding='same')(x)
        x = layers.Activation('elu')(x)
        x = layers.Add()([x, x0])
        x = layers.AveragePooling2D()(x)
        return x

    def _upsample_blodk(self, x, filters):
        x0 = x
        x = layers.Activation('elu')(x)
        x = layers.Conv2DTranspose(filters, 7, padding='same', data_format='channels_last')(x)
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

        x = layers.Conv2D(64, 7, padding='same')(x)
        x = self._downsample_block(x, 64)
        x = self._downsample_block(x, 64)
        x = self._downsample_block(x, 64)
        # x = self._downsample_block(x, 128)

        z_mean = layers.Conv2D(1, 16, padding='same', activation='elu')(x)
        z_mean = layers.Conv2D(1, 1, padding='same', name="z_mean")(z_mean)
        z_mean = layers.Reshape((16, 16))(z_mean)

        z_log_var = layers.Conv2D(1, 16, padding='same', activation='elu')(x)
        z_log_var = layers.Conv2D(1, 1, padding='same', name="z_log_var")(z_log_var)
        z_log_var = layers.Reshape((16, 16))(z_log_var)

        z = Sampling()([z_mean, z_log_var])

        return tf.keras.Model(inputs, [z_mean, z_log_var, z])

    def _get_decoder(self):
        inputs = layers.Input(shape=(16, 16))
        x = layers.Reshape((16, 16, 1))(inputs)
        x = self._upsample_blodk(x, 64)
        x = self._upsample_blodk(x, 64)
        x = self._upsample_blodk(x, 64)
        # x = self._upsample_blodk(x, 128)

        x = layers.Cropping2D((3, 0))(x)
        x = layers.ZeroPadding2D(((0, 0), (1, 0)))(x)

        x = layers.Conv2DTranspose(64, 7, padding='same')(x)
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
    vae = SpectrogramVAE(normalization_layer=layers.Normalization())
    vae.encoder.summary()
    vae.decoder.summary()

    o = vae.encoder(get_test_data())
    # mcnn = STFTInverter()
    # mcnn.mcnn.summary()
    #
    # ds = get_test_data()
    # out = mcnn.mcnn(ds)
    #
    # print(out)