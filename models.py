import keras
import tensorflow as tf
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the STFT."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def kl_divergence(z_mean, z_log_var):
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))


def reconstruction_loss(real, reconstruction):
    return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(real, reconstruction), axis=(1, 2)))


def spectral_convergence_loss(real, predicted):
    return tf.reduce_mean(
        tf.math.reduce_euclidean_norm(tf.math.abs(real) - tf.math.abs(predicted)) / tf.math.reduce_euclidean_norm(tf.math.abs(real))
    )


class SpectrogramVAE(tf.keras.Model):

    def __init__(self, normalization_layer, spectrogram_shape=(12, 128, 1), latent_dim=16, **kwargs):
        super(SpectrogramVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.spectrogram_shape = spectrogram_shape
        self.normalization_layer = normalization_layer

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

    def _downsample_block(self, x, filters):
        x0 = x
        x = layers.Activation('tanh')(x)
        x = layers.Conv2D(filters, 3, padding='same', data_format='channels_last')(x)
        x = layers.Activation('tanh')(x)
        x = layers.Conv2D(filters, 1, padding='same')(x)
        x = layers.Add()([x, x0])
        x = layers.AveragePooling2D()(x)
        return x

    def _upsample_blodk(self, x, filters):
        x0 = x
        x = layers.Activation('tanh')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same', data_format='channels_last')(x)
        x = layers.Activation('tanh')(x)
        x = layers.Conv2DTranspose(filters, 1, padding='same')(x)
        x = layers.Add()([x, x0])
        x = layers.UpSampling2D()(x)
        return x

    def _get_encoder(self):
        inputs = layers.Input(shape=self.spectrogram_shape)
        x = self.normalization_layer(inputs)
        x = layers.Conv2D(32, 3, padding='same')(x)
        x = self._downsample_block(x, 32)
        x = self._downsample_block(x, 32)
        x = layers.Flatten()(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        return tf.keras.Model(inputs, [z_mean, z_log_var, z])

    def _get_decoder(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(3 * 32 * 32)(inputs)
        x = layers.Reshape((3, 32, 32))(x)
        x = self._upsample_blodk(x, 32)
        x = self._upsample_blodk(x, 32)
        x = layers.Conv2DTranspose(32, 3, padding='same')(x)
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


if __name__ == '__main__':
    vae = SpectrogramVAE(normalization_layer=layers.Normalization())
    vae.encoder.summary()
    vae.decoder.summary()
