import tensorflow as tf


def kl_divergence(z_mean, z_log_var):
    coefficient = 0.5
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss *= coefficient
    return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))


def reconstruction_loss(real, reconstruction):
    return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(real, reconstruction), axis=(1, 2)))


def spectral_convergence_loss(real, predicted):
    return tf.reduce_mean(
        tf.norm(real - predicted, ord='fro', axis=[1, 2]) / tf.norm(real, ord='fro', axis=[1, 2])
    )


def log_scale_stft_magnitude_loss(real, predicted):
    epsilon = 1e-10
    return tf.reduce_mean(
        tf.norm(
            tf.math.log(real + epsilon) - tf.math.log(predicted + epsilon), ord=1, axis=[1, 2])
    )

def stft_magnitude_loss(real, predicted):
    return tf.reduce_mean(
        tf.norm(real - predicted, ord=1, axis=[1, 2])
    )