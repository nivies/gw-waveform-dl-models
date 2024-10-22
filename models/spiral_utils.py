import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import keras
from keras import layers
from keras import regularizers



class SpiralLayer(Layer):
    def __init__(self, input_dim, **kwargs):
        super(SpiralLayer, self).__init__(**kwargs)
        self.a = self.add_weight(name='a_weight', shape=(input_dim,), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='b_weight', shape=(input_dim,), initializer='random_normal', trainable=True)
        self.c = self.add_weight(name='c_weight', shape=(input_dim,), initializer='random_normal', trainable=True)
        self.d = self.add_weight(name='d_weight', shape=(input_dim,), initializer='random_normal', trainable=True)
        self.input_dim = input_dim

    def call(self, inputs):
        x = inputs
        spiral_x = (self.a + x * self.b) * tf.sin(x * 2 * tf.constant(np.pi))
        spiral_y = (self.c + x * self.d) * tf.cos(x * 2 * tf.constant(np.pi))
        output = tf.concat([spiral_x, spiral_y], axis=-1)
        return output

    def get_config(self):
        config = super(SpiralLayer, self).get_config()
        config.update({"input_dim": self.input_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def declare_NN(in_out_shapes, latent_dim):
    pars = keras.Input(in_out_shapes['input_shape'])
    wvfs = keras.Input(in_out_shapes['output_shape'])

    x_p = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(pars)
    x_p = layers.Dense(latent_dim, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform',  activity_regularizer = regularizers.L1L2(l1=1e-5, l2=1e-4))(x_p)
    inp_rep = SpiralLayer(latent_dim)(x_p)

    x_w = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(wvfs)
    x_w = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x_w)
    x_w = layers.Dense(latent_dim, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform', activity_regularizer = regularizers.L1L2(l1=1e-5, l2=1e-4))(x_w)

    wv_rep = SpiralLayer(latent_dim)(x_w)

    y_w = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(inp_rep)
    y_w = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y_w)
    y_w = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y_w)
    opt = layers.Dense(in_out_shapes['output_shape'], activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y_w)

    pars_encoder = keras.Model(pars, inp_rep)
    wv_encoder = keras.Model(wvfs, wv_rep)
    decoder = keras.Model(inp_rep, opt)

    return pars_encoder, wv_encoder, decoder


class SpiralAutoEncoder(keras.Model):

    def __init__(self, par_encoder, wv_encoder, decoder, lbd, **kwargs):
        super(SpiralAutoEncoder, self).__init__(**kwargs)
        self.par_encoder = par_encoder
        self.wv_encoder = wv_encoder
        self.decoder = decoder
        self.lbd = lbd

        inp = par_encoder.input
        enc = par_encoder.output

        self.model = keras.Model(inp, decoder(enc))

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.encoding_loss_tracker = keras.metrics.Mean(name="encoding_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.encoding_loss_tracker,
        ]

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            pars, waves = data

            par_rep = self.par_encoder(pars)
            wv_rep = self.wv_encoder(waves)
            opt = self.decoder(par_rep)

            encoding_loss = keras.losses.MeanAbsoluteError()(par_rep, wv_rep)
            reconstruction_loss = keras.losses.MeanAbsoluteError()(opt, waves)

            total_loss = (1.0 - self.lbd) * reconstruction_loss + self.lbd * encoding_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.encoding_loss_tracker.update_state(encoding_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "encoding_loss": self.encoding_loss_tracker.result(),
        }
    
    def test_step(self, data):

        pars, waves = data

        par_rep = self.par_encoder(pars)
        wv_rep = self.wv_encoder(waves)
        opt = self.decoder(par_rep)

        encoding_loss = keras.losses.MeanAbsoluteError()(par_rep, wv_rep)
        reconstruction_loss = keras.losses.MeanAbsoluteError()(opt, waves)

        total_loss = (1.0 - self.lbd) * reconstruction_loss + self.lbd * encoding_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "encoding_loss": encoding_loss,
        }

    def call(self, data):

        emb = self.par_encoder(data)
        return self.decoder(emb)