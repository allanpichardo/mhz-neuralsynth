# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Melspectrogram to wav by GL algorithm."""
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
import librosa


class TFGriffinLim(tf.keras.layers.Layer):
    """GL algorithm."""

    def __init__(self, norm_mean, norm_variance, n_fft=2048, n_mels=128, sr=16000, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self._n_iters = 60
        self.norm_mean = norm_mean.numpy()
        self.norm_variance = norm_variance.numpy()
        
        self.setup_stats()

    def setup_stats(self):
        scaler = StandardScaler()
        scaler.mean_ = self.norm_mean
        scaler.scale_ = self.norm_variance
        self._scaler = scaler

    def _de_normalization(self, mel_spectrogram):
        return self._scaler.inverse_transform(mel_spectrogram)

    @tf.function
    def _build_mel_basis(self):
        return librosa.filters.mel(self.sr,
                                   self.n_fft,
                                   n_mels=self.n_mels,
                                   fmin=0.0,
                                   fmax=None)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def _mel_to_linear(self, mel_spectrogram):
        _inv_mel_basis = tf.linalg.pinv(self._build_mel_basis())
        return tf.math.maximum(1e-10, tf.matmul(_inv_mel_basis, tf.transpose(mel_spectrogram, (1, 0))))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.complex64)])
    def _invert_spectrogram(self, spectrogram):
        '''
        spectrogram: [t, f]
        '''
        spectrogram = tf.expand_dims(spectrogram, 0)
        inversed = tf.signal.inverse_stft(
            spectrogram,
            self.n_fft,
            self.n_fft // 4,
            self.n_fft
        )
        return tf.squeeze(inversed, 0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def run_convert(self, mel_spectrogram):
        spectrogram = self._mel_to_linear(tf.pow(10.0, mel_spectrogram))
        spectrogram = tf.transpose(spectrogram, (1, 0))
        spectrogram = tf.cast(spectrogram, dtype=tf.complex64)
        best = tf.identity(spectrogram)

        for _ in tf.range(self._n_iters):
            best = self._invert_spectrogram(spectrogram)
            estimate = tf.signal.stft(
                best,
                self.n_fft,
                self.n_fft // 4,
                self.n_fft
            )
            phase = estimate / tf.cast(tf.maximum(1e-10, tf.abs(estimate)), tf.complex64)
            best = spectrogram * phase

        y = tf.math.real(self._invert_spectrogram(best))
        return y

    def call(self, mel_spectrogram):
        mel_spectrogram = self._de_normalization(mel_spectrogram)
        return self.run_convert(mel_spectrogram)
