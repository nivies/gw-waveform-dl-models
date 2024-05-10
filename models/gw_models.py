from base.base_model import BaseModel
from models.gw_latent_mappers import UMAPMapper
from models.parametric_umap import ParametricUMAP, Embedder
from keras import layers
import keras
import os


class MLP(BaseModel):
    def __init__(self, config, data_loader, latent_mapper_dim = None):
        super(MLP, self).__init__(config)
        self.latent_dim = latent_mapper_dim
        self.in_out_shapes = data_loader.in_out_shapes
        self.build_model()

    def build_model(self):

        params = keras.Input(self.in_out_shapes['input_shape'])

        x = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(params) 
        x = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(512, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(1024, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x) 
        x = layers.Dense(1024, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(1024, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(1024, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(1024, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)

        if self.latent_dim:
            opt = layers.Dense(self.latent_dim)(x)
        else:
            opt = layers.Dense(self.in_out_shapes['output_shape'])(x)

        self.model = keras.Model(params, opt)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)

        self.model.compile(optimizer = optimizer,
                           loss = 'mae'
                           )
        
class RegularizedAutoEncoder(BaseModel):
    def __init__(self, config, out_shape):
        super(RegularizedAutoEncoder, self).__init__(config)
        self.latent_dim = config.model.latent_dim
        self.out_shape = out_shape
        self.build_model()

    def build_model(self):
        
        inp = keras.Input(self.out_shape)

        x = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(inp)
        x = layers.Dense(units=512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        x = layers.Reshape(target_shape=(512, 1))(x)
        x = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=512, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=512, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=256, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        x = layers.Dense(units=512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        enc = layers.Dense(units=self.latent_dim, name = 'encoding')(x)

        y = layers.Dense(units=512, activation='leaky_relu', kernel_initializer='glorot_uniform')(enc)
        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)
        y = layers.Reshape(target_shape=(32, 32))(y)
        y = layers.Conv1DTranspose(filters=128, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.UpSampling1D(size=2)(y)
        y = layers.Conv1DTranspose(filters=128, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.Conv1DTranspose(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.UpSampling1D(size=2)(y)
        y = layers.Conv1DTranspose(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.Conv1DTranspose(filters=512, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.UpSampling1D(size=2)(y)
        y = layers.Conv1DTranspose(filters=512, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.Conv1DTranspose(filters=2, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.Flatten()(y)
        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)
        
        dec = layers.Dense(units=self.out_shape, activation='linear', name = 'reconstruction')(y)

        self.autoencoder = keras.Model(inp, [enc, dec])
        self.encoder = keras.Model(inp, enc)
        self.decoder = keras.Model(enc, dec)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)
        self.autoencoder.compile(optimizer = optimizer,
                                loss_weights=[self.config.model.reg_weight, 1-self.config.model.reg_weight],
                                loss = {'encoding' : 'mean_absolute_error', 'reconstruction' : 'mean_absolute_error'}
                                )

class AutoEncoder(BaseModel):
    def __init__(self, config, out_shape):
        super(AutoEncoder, self).__init__(config)
        self.latent_dim = config.model.latent_dim
        self.out_shape = out_shape
        self.build_model()

    def build_model(self):
        
        inp = keras.Input(self.out_shape)

        x = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(inp)
        x = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        x = layers.Reshape(target_shape=(1024, 1))(x)
        x = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=64, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=512, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=512, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=256, kernel_size=5, dilation_rate=2, padding='causal', activation='leaky_relu')(x)
        x = layers.AveragePooling1D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        x = layers.Dense(units=512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        enc = layers.Dense(units=self.latent_dim, name = 'encoding')(x)

        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(enc)
        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)
        y = layers.Reshape(target_shape=(32, 32))(y)
        y = layers.Conv1DTranspose(filters=128, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.UpSampling1D(size=2)(y)
        y = layers.Conv1DTranspose(filters=128, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.Conv1DTranspose(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.UpSampling1D(size=2)(y)
        y = layers.Conv1DTranspose(filters=256, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.Conv1DTranspose(filters=512, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.UpSampling1D(size=2)(y)
        y = layers.Conv1DTranspose(filters=512, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.Conv1DTranspose(filters=2, kernel_size=5, padding='same', dilation_rate=2, activation='leaky_relu')(y)
        y = layers.Flatten()(y)
        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)
        
        dec = layers.Dense(units=self.out_shape, activation='linear', name = 'reconstruction')(y)

        self.autoencoder = keras.Model(inp, dec)
        self.encoder = keras.Model(inp, enc)
        self.decoder = keras.Model(enc, dec)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)
        self.autoencoder.compile(optimizer = optimizer,
                                loss_weights=[self.config.model.reg_weight, 1-self.config.model.reg_weight],
                                loss = 'mean_absolute_error'
                                )


class RegularizedAutoEncoderGenerator(BaseModel):
    def __init__(self, config, data_loader, inference = False):
        super(RegularizedAutoEncoderGenerator, self).__init__(config)
        
        self.in_out_shapes = data_loader.in_out_shapes
        self.inference = inference

        self.mapper = UMAPMapper(config).mapper
        if self.inference:
            self.embedder = Embedder(config, self.in_out_shapes['input_shape'], config.model.latent_dim)
        else:
            self.embedder = ParametricUMAP(config, data_loader)
        self.autoencoder = RegularizedAutoEncoder(config, self.in_out_shapes['output_shape'])
        self.build_model()

    def build_model(self):

        inp = keras.Input(self.in_out_shapes['input_shape'])
        emb = self.embedder.embedder(inp)
        lat = self.mapper(emb)
        opt = self.autoencoder.decoder(lat)

        self.model = keras.Model(inp, opt)

class MappedAutoEncoderGenerator(BaseModel):
    def __init__(self, config, data_loader):
        super(MappedAutoEncoderGenerator, self).__init__(config)
        
        self.in_out_shapes = data_loader.in_out_shapes

        self.mapper = MLP(config, data_loader, self.config.model.latent_dim).model
        self.autoencoder = AutoEncoder(config, self.in_out_shapes['output_shape'])
        self.build_model()

    def build_model(self):

        inp = keras.Input(self.in_out_shapes['input_shape'])
        lat = self.mapper(inp)
        opt = self.autoencoder.decoder(lat)

        self.model = keras.Model(inp, opt)