import tensorflow as tf
import keras
from keras import layers
from models.gw_test_models import stack_residual_blocks

'''
File for architecture definition and internal logic coding of conditional variational autoencoder.
'''

def declare_encoder(inputs: tf.Tensor, model_id: str, config) -> tf.Tensor:

    if model_id == "conv":
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
        return layers.Dense(1024, activation="leaky_relu", kernel_initializer = 'glorot_normal')(x)

    elif model_id == "0":
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(inputs)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        return layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)

    elif model_id == "1":
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(inputs)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        return layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
    
    elif model_id == "2":
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(inputs)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        return layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
    
    elif model_id == "3":
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(inputs)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
        return layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(x)
    
    elif model_id == "deep_encoder":

        n_layers = config.model.deep.encoder_n_layers
        units = config.model.deep.encoder_n_units

        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(inputs) 

        for _ in range(int(n_layers)):
            x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)

        return layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
    
    elif model_id == "deep_residual_encoder":

        n_layers = config.model.deep.encoder_n_layers
        units = config.model.deep.encoder_n_units
        res_layers = []

        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(inputs) 

        for ct in range(int(n_layers/2)):
            
            x = layers.Dense(units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
            if ct % 10 == 0:
                res_layers.append(tf.identity(x))

        for ct in range(int(n_layers/2)):

            if ct % 10 == 0:
                x = layers.Dense(units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x + res_layers.pop(-1))

            else:
                x = layers.Dense(units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)

        return layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
    
    elif model_id == "deep_block_residual_encoder":

        
        n_blocks = config.model.deep.mapper_n_blocks
        layers_per_block = config.model.deep.mapper_layers_per_block
        units = config.model.deep.mapper_n_units
        dense_shortcut = config.model.deep.mapper_dense_shortcut 

        return stack_residual_blocks(x = inputs, n_blocks = n_blocks, n_neurons = units, layers_per_block = layers_per_block, dense_shortcut = dense_shortcut) 

    
    else:
        raise ValueError(f"Model id supplied ({model_id}) is not implemented.")

def declare_decoder(z_cond: tf.Tensor, model_id: str, config) -> tf.Tensor:

    if model_id == "conv":
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
        return layers.Flatten()(y)

    elif model_id == "0":
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(z_cond)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        return layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)

    elif model_id == "1":
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(z_cond)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        return layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
    
    elif model_id == "2":
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(z_cond)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        return layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
    
    elif model_id == "3":
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(z_cond)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        y = layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
        return layers.Dense(512, activation = "relu", kernel_initializer = "glorot_uniform")(y)
    
    elif model_id == "deep_decoder":

        n_layers = config.model.deep.decoder_n_layers
        units = config.model.deep.decoder_n_units

        y = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(z_cond) 

        for _ in range(int(n_layers)):
            y = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y)

        return layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y)
    
    elif model_id == "deep_residual_decoder":

        n_layers = config.model.deep.decoder_n_layers
        units = config.model.deep.decoder_n_units
        res_layers = []

        y = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(z_cond) 

        for ct in range(int(n_layers/2)):
            
            y = layers.Dense(units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y)
            if ct % 10 == 0:
                res_layers.append(tf.identity(y))

        for ct in range(int(n_layers/2)):

            if ct % 10 == 0:
                y = layers.Dense(units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y + res_layers.pop(-1))

            else:
                y = layers.Dense(units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y)

        return layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y)
    
    elif model_id == "deep_block_residual_decoder":

        n_blocks = config.model.deep.mapper_n_blocks
        layers_per_block = config.model.deep.mapper_layers_per_block
        units = config.model.deep.mapper_n_units
        dense_shortcut = config.model.deep.mapper_dense_shortcut 

        return stack_residual_blocks(x = z_cond, n_blocks = n_blocks, n_neurons = units, layers_per_block = layers_per_block, dense_shortcut = dense_shortcut) 

    
    
    else:
        raise ValueError(f"Model id supplied ({model_id}) is not implemented.")

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

def cVAE_NN_declaration(in_out_shapes: dict, latent_dim: int, encoder_id: str, decoder_id: str, config) -> keras.Model:

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

    x = declare_encoder(inputs, encoder_id, config)

    z_mean = layers.Dense(latent_dim, kernel_initializer = 'glorot_normal', name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, kernel_initializer = 'glorot_normal', name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    z_cond = layers.Concatenate(axis=1)([z, par_inputs])

    y = declare_decoder(z_cond, decoder_id, config)

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
    
    def get_config(self):
        # Return the configuration for serialization
        config = super(cVAE, self).get_config()
        config.update({
            "encoder": self.encoder,
            "decoder": self.decoder
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Rebuild the object from configuration
        encoder = config.pop("encoder")
        decoder = config.pop("decoder")
        return cls(encoder=encoder, decoder=decoder, **config)
    