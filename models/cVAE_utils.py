import tensorflow as tf
import keras
from keras import layers

'''
File for architecture definition and internal logic coding of conditional variational autoencoder.
'''


class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the latent space vector.

    Input parameters:
    -----------------

    inputs: tuple
        z_mean and z_log_var learnt from the encoder network.
    -----------------

    Output parameters:
        z_sampled: A sample from the learned distribution.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def cVAE_NN_declaration(in_out_shapes, latent_dim):

    '''
    Defines the encoder and decoder architectures for the cVAE.

    Input parameters:
    -----------------

    in_out_shapes: data_loader built-in dictionary
        Input and output shapes for the model.

    latent_dim: int
        Latent dimensionality
    -----------------

    Output parameters:
        Encoder and decoder keras' models.
    '''

    input_shape = in_out_shapes['input_shape']
    output_shape = in_out_shapes['output_shape']

    wave_inputs = keras.Input(shape=output_shape)
    par_inputs = keras.Input(shape=input_shape)
    inputs = layers.Concatenate(axis=1)([wave_inputs, par_inputs])
    x = layers.Dense(1024, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(inputs)
    x = layers.Reshape((1024, 1))(x)
    x = layers.Conv1D(64, 7, activation="leaky_relu", padding="causal", dilation_rate = 2)(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(64, 7, activation="leaky_relu", padding="causal", dilation_rate = 2)(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(128, 7, activation="leaky_relu", padding="causal", dilation_rate = 2)(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(128, 7, activation="leaky_relu", padding="causal", dilation_rate = 2)(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(256, 7, activation="leaky_relu", padding="causal", dilation_rate = 2)(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(256, 7, activation="leaky_relu", padding="causal", dilation_rate = 2)(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="leaky_relu", kernel_initializer = 'glorot_normal')(x)
    x = layers.Dense(1024, activation="leaky_relu", kernel_initializer = 'glorot_normal')(x)
    z_mean = layers.Dense(latent_dim, kernel_initializer = 'glorot_normal', name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, kernel_initializer = 'glorot_normal', name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    z_cond = layers.Concatenate(axis=1)([z, par_inputs])

    y = layers.Dense(1024, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(z_cond)
    y = layers.Dense(16*256, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y)
    y = layers.Reshape((16, 256))(y)
    y = layers.Conv1DTranspose(256, 7, activation = 'leaky_relu', padding = 'same', dilation_rate = 2)(y)
    y = layers.UpSampling1D(2)(y)
    y = layers.Conv1DTranspose(256, 7, activation = 'leaky_relu', padding = 'same', dilation_rate = 2)(y)
    y = layers.UpSampling1D(2)(y)
    y = layers.Conv1DTranspose(128, 7, activation = 'leaky_relu', padding = 'same', dilation_rate = 2)(y)
    y = layers.UpSampling1D(2)(y)
    y = layers.Conv1DTranspose(128, 7, activation = 'leaky_relu', padding = 'same', dilation_rate = 2)(y)
    y = layers.UpSampling1D(2)(y)
    y = layers.Conv1DTranspose(64, 7, activation = 'leaky_relu', padding = 'same', dilation_rate = 2)(y)
    y = layers.UpSampling1D(2)(y)
    y = layers.Conv1DTranspose(64, 7, activation = 'leaky_relu', padding = 'same', dilation_rate = 2)(y)
    y = layers.UpSampling1D(2)(y)
    y = layers.Conv1DTranspose(1, 7, activation = 'leaky_relu', padding = 'same', dilation_rate = 2)(y)
    y = layers.Flatten()(y)
    output = layers.Dense(output_shape, kernel_initializer = 'glorot_uniform')(y)


    encoder = keras.Model([wave_inputs, par_inputs], [z_mean, z_log_var, z], name="encoder")
    decoder = keras.Model([z, par_inputs], output, name = 'decoder')

    return encoder, decoder

class cVAE(keras.Model):

    '''
    Defines the training logic of the conditional variational autoencoder. Inherits from keras.Model. Defined to be called with two inputs in the correct 
    order: [Outputs, Conditions]. (For calling only the decoder, the input order is inversed).

    Input parameters:
    -----------------

    encoder: keras.Model
        Encoder model.

    decoder: keras.Model
        Decoder model.
    -----------------
    '''

    def __init__(self, encoder, decoder, **kwargs):
        super(cVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            [waves, conditions], waves_ = data
            z_mean, z_log_var, z = self.encoder([waves, conditions])
            reconstruction = self.decoder([z, conditions])
            reconstruction_loss = keras.losses.MeanSquaredError()(waves, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):

        [waves, conditions], waves_ = data
        z_mean, z_log_var, z = self.encoder([waves, conditions])
        reconstruction = self.decoder([z, conditions])
        mse = keras.metrics.mean_squared_error(waves, reconstruction)
        reconstruction_loss = keras.losses.MeanSquaredError()(waves, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, data):

        waves, conditions = data
        z_mean, z_log_var, z = self.encoder([waves, conditions])
        return self.decoder([z, conditions])
    