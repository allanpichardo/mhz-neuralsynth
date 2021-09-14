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


class SampleVAE(tf.keras.Model):

    def __init__(self, vector_size=128, latent_dim=16, **kwargs):
        super(SampleVAE, self).__init__(**kwargs)
        self.vector_size = vector_size
        self.latent_dim = latent_dim
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.prediction_loss_tracker = tf.keras.metrics.Mean(name="prediction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        self.encoder, filters, dilation = self._get_encoder()
        self.decoder = self._get_decoder(filters=filters, dilation=dilation)

    def _get_encoder(self):
        current = self.vector_size
        filters = 16
        dilation = 1

        inputs = layers.Input(shape=(self.vector_size,))
        x = layers.Reshape((self.vector_size, 1))(inputs)

        while current > self.latent_dim:
            x = layers.Conv1D(filters, 3, dilation_rate=dilation, padding='same', activation='tanh')(x)
            x = layers.MaxPooling1D()(x)
            current //= 2
            filters *= 2
            dilation *= 2

        z_mean = layers.Conv1D(1, 1, padding='same', activation=None, name="z_mean")(x)
        z_mean = layers.Flatten()(z_mean)

        z_log_var = layers.Conv1D(1, 1, padding='same', activation=None, name="z_log_var")(x)
        z_log_var = layers.Flatten()(z_log_var)

        z = Sampling()([z_mean, z_log_var])

        return tf.keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z]), filters // 2, dilation // 2

    def _get_decoder(self, filters=64, dilation=4):
        current = self.latent_dim
        filters = filters
        dilation = dilation

        inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Reshape((self.latent_dim, 1))(inputs)

        while current < self.vector_size:
            x = layers.Conv1DTranspose(filters, 3, dilation_rate=dilation, padding='same', activation='tanh')(x)
            x = layers.UpSampling1D()(x)
            current *= 2
            filters //= 2
            dilation //= 2

        x = layers.Conv1DTranspose(1, 1, padding='same', activation=None)(x)
        x = layers.Reshape((self.vector_size,))(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def call(self, inputs, training=None, mask=None):
        u, v, z = self.encoder(inputs)
        y_pred = self.decoder(z)
        return y_pred

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.prediction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
            y = data[1]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            completion = self.decoder(z)

            prediction_loss = tf.keras.losses.Huber()(y, completion)

            coefficient = 0.0001
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * coefficient
            total_loss = kl_loss + prediction_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


if __name__=='__main__':
    vae = SampleVAE()
    vae.encoder.summary()
    vae.decoder.summary()
    vae.summary()