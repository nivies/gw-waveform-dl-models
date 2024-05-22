from base.base_model import BaseModel
from models.gw_latent_mappers import UMAPMapper
from models.parametric_umap import ParametricUMAP, Embedder
from keras import layers
import keras
from models.cVAE_utils import *
from models.gw_test_models import *

'''
File for declaring the main architectures for the DL-based GW modelling neural networks. Every class inherits from the BaseModel
class defined in the base folder.
'''

class MLP(BaseModel):

    '''
    Class for declaring a dense neural network. It can be called to be used directly as a GW model, or from the MappedAutoencoder 
    class in order to map the latent space. It is built entirely from the data_loader instance and the config .json
    file. If called from the MappedAutoencoder class, the latent dimension is given in the call. The class defines the model and compiles it.
    

    Input arguments:
    ----------------
    config: dict
        Dictionary built from the configuration .json file.

    data_loader: data_loader class instance
        data_loader instance called. Information for the calling contained in the configuration .json file.

    latent_mapper_dim: int
        Value for the latent dimension in case of instancing for latent space mapping for the MappedAutoencoder class.

    ----------------

    Defined model stored in the .model method.
    '''


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

    '''
    Class for the definition of the 1D convolutional autoencoder architecture with a regularization in the latent space. Called from the 
    RegularizedAutoEncoderGenerator class. It defines a preset autoencoder architecture with the necessary logic for the 
    regularization implementation.

    Input arguments:
    ----------------

    config: dict
        Dict built from the configuration .json file.

    out_shape: int
        Parameter controlling the dimensionality of the data to fit. Particularly, the size of the waveform vectors.
    ----------------

    Autoencoder built stored in .autoencoder method. Similarly, the encoder and decoder are stored in the .encoder and the
    .decoder methods.
    '''


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

    '''
    Class for the definition of a standard autoencoder architecture. This class is to be called from the MappedAutoEncoderGenerator class.
    It declares a predefined 1D convolutional autoencoder architecture that will fit the waveforms.
    
    Input arguments:
    ----------------

    config: dict
        Dict built from the configuration .json file.

    out_shape: int
        Parameter controlling the dimensionality of the data to fit. Particularly, the size of the waveform vectors.
    ----------------

    Autoencoder built stored in .autoencoder method. Similarly, the encoder and decoder are stored in the .encoder and the
    .decoder methods.
    '''

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

    '''
    Class for calling all the necessary models (UMAP embedder, UMAP mapper and regularized autoencoder) and ensembling the final
    regularized autoencoder GW generation model.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary built from the configuration .json file.

    data_loader: data_loader class instance
        data_loader instance called for the particular problem. All information for this class' calling is contained in the configuration file.

    inference: bool
        Parameter for defining the model in inference or training mode. If True, it skips the building of the Parametric UMAP class, speeding 
        up the loading time.
    -----------------

    Generator model built can be called from the .model method. Every network that composes the full model can also be called:

    embedder    : UMAP embedder class.
    mapper      : Mapper network from the UMAP embedder to the latent space of the regularized autoencoder.
    autoencoder : Regularized autoencoder class.
    '''

    def __init__(self, config, data_loader, test = False, inference = False):
        super(RegularizedAutoEncoderGenerator, self).__init__(config)
        
        self.in_out_shapes = data_loader.in_out_shapes
        self.inference = inference

        if test:

            self.mapper = UMAPMapper_test(config).mapper
            if self.inference:
                self.embedder = Embedder_test(config, self.in_out_shapes['input_shape'], config.model.latent_dim)
            else:
                self.embedder = ParametricUMAP(config, data_loader, test = True)
            self.autoencoder = RegularizedAutoEncoder_test(config, self.in_out_shapes['output_shape'])

        else:

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

    '''
    Class for calling all the necessary models (mapper and autoencoder) and ensembling the final mapped autoencoder
    GW generation model.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary built from the configuration .json file.

    data_loader: data_loader class instance
        data_loader instance called for the particular problem. All information for this class' calling is contained in the configuration file.
    -----------------

    Generator model built can be called from the .model method. Every network that composes the full model can also be called:

    mapper      : Mapper network from the input parameters to the latent space of the autoencoder.
    autoencoder : Mapped autoencoder class.
    '''

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

class cVAEGenerator(BaseModel):

    '''
    Class for defining the cVAE model and include it in the project's pipeline.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary built from the configuration .json file.

    data_loader: data_loader class instance
        data_loader instance called for the particular problem. All information for this class' calling is contained in the configuration file.
    -----------------

    Generator model built can be called from the .model method. Every network that composes the full model can also be called:

    encoder: Encoder network.
    decoder: Decoder network.
    model  : cVAE model.
    '''

    def __init__(self, config, data_loader, inference = False):
        super(cVAEGenerator, self).__init__(config)
        
        self.in_out_shapes = data_loader.in_out_shapes

        self.build_model()

    def build_model(self):

        self.encoder, self.decoder = cVAE_NN_declaration(self.in_out_shapes, self.config.model.latent_dim)

        self.model = cVAE(self.encoder, self.decoder)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)
        self.model.compile(optimizer = optimizer,
                           loss = 'mae'
                           )