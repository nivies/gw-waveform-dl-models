from base.base_model import BaseModel
import keras
from keras import layers
from utils.loss import *

def get_pca_model_from_id(input, model_id):
        
    if model_id == '0' or '2':
        units = 128

    elif model_id == '1' or '3':
        units = 512

    
    if model_id == '0' or '1':
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(input) 
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        return layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
    
    elif model_id == '2' or '3':

        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(input) 
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        return layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)



def get_ae_model_from_id(input, model_id, latent_dimension, output_dimension, config):

    # Improve on how the arguments are passed: most of this stuff is in the config file

    units = 512

    
    if model_id == '0':
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(input) 
        enc = layers.Dense(units = latent_dimension, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform', name = 'latent_components')(x)
        dec = layers.Dense(units = output_dimension, kernel_initializer = 'glorot_uniform', name = 'output')(enc)
    
    elif model_id == '1':

        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(input) 
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        enc = layers.Dense(units = latent_dimension, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform', name = 'latent_components')(x)
        y = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(enc)
        y = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y)
        dec = layers.Dense(units = output_dimension, kernel_initializer = 'glorot_uniform', name = 'output')(y)

    elif model_id == "l1_regularized":
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(input) 
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        enc = layers.Dense(units = latent_dimension, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform', name = 'latent_components', kernel_regularizer = keras.regularizers.L1(l1 = config.model.reg_weight))(x)
        y = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(enc)
        y = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y)
        dec = layers.Dense(units = output_dimension, kernel_initializer = 'glorot_uniform', name = 'output')(y)

    elif model_id == "custom_regularized":
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(input) 
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        x = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        enc = layers.Dense(units = latent_dimension, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform', name = 'latent_components', kernel_regularizer = ComponentWiseRegularizer(coef = config.model.reg_weight))(x)
        y = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(enc)
        y = layers.Dense(units = units, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(y)
        dec = layers.Dense(units = output_dimension, kernel_initializer = 'glorot_uniform', name = 'output')(y)

    else:
        raise Exception("Autoencoder ID not defined!")
        
    return enc, dec

class ConvBlock1D_test(BaseModel):

    '''
    Convolutional block model instance.

    Defines a 1D convolutional block consisting in 3 repetitions of:

    1. 1D convolutional layer.
    2. 1D Spatial Dropout.
    3. 1D Upsampling layer.
    4. Leaky ReLU activation.

    Input and output dimensions are set equal in order to implement a residual network with Add layers. This block is set
    to be called by the UMAP mapper class if the argument model.mapper_architecture is set to convolutional.

    Input arguments:
    ----------------
    config: dict
        Config dict loaded from config .json file.

    input_shape: int 
        Number of input and output dimensions of convolutional block.
    ----------------
    Declared model is stored in the .conv_block method.

    '''

    def __init__(self, config, input_shape):
        super(ConvBlock1D_test, self).__init__(config)

        self.input_shape = input_shape
        self.build_model()
    
    def build_model(self):

        dropout_rate = self.config.model.mapper_dropout

        inp = keras.Input(self.input_shape)
        x = layers.Reshape((self.input_shape, 1))(inp)

        x = layers.Conv1D(filters = 512, kernel_size = 3, padding = "same")(x)
        x = layers.SpatialDropout1D(rate = dropout_rate)(x)
        x = layers.UpSampling1D(size = 2)(x)
        x = layers.LeakyReLU()(x)
                
        x = layers.Conv1D(filters = 512, kernel_size = 3, padding = "same")(x)
        x = layers.SpatialDropout1D(rate = dropout_rate)(x)
        x = layers.UpSampling1D(size = 2)(x)
        x = layers.LeakyReLU()(x)

                
        x = layers.Conv1D(filters = 512, kernel_size = 3, padding = "same")(x)
        x = layers.SpatialDropout1D(rate = dropout_rate)(x)
        x = layers.UpSampling1D(size = 2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Flatten()(x)

        x = layers.Dense(self.input_shape)(x)

        self.conv_block = keras.Model(inp, x)

class RegularizedAutoEncoder_test(BaseModel):

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
        super(RegularizedAutoEncoder_test, self).__init__(config)
        self.latent_dim = config.model.latent_dim
        self.out_shape = out_shape
        self.build_model()

    def build_model(self):
        
        inp = keras.Input(self.out_shape)

        x = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(inp)
        x = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        x = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        enc = layers.Dense(units=self.latent_dim, name = 'encoding')(x)

        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(enc)
        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)
        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)
        
        dec = layers.Dense(units=self.out_shape, activation='linear', name = 'reconstruction')(y)

        self.autoencoder = keras.Model(inp, [enc, dec])
        self.encoder = keras.Model(inp, enc)
        self.decoder = keras.Model(enc, dec)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)

        # if self.config.model.loss == "overlap":

        #     self.autoencoder.compile(optimizer = optimizer,
        #                         loss_weights=[self.config.model.reg_weight, 1-self.config.model.reg_weight],
        #                         loss = {'encoding' : 'mean_absolute_error', 'reconstruction' : ovlp_mae_loss},
        #                         metrics = {'encoding' : 'mean_absolute_error', 'reconstruction' : [overlap, 'mean_absolute_error']}
        #                         )
        # else:

        self.autoencoder.compile(optimizer = optimizer,
                            loss_weights=[self.config.model.reg_weight, 1-self.config.model.reg_weight],
                            loss = {'encoding' : 'mean_absolute_error', 'reconstruction' : 'mean_absolute_error'}
                            )      

class Embedder_test(BaseModel):

    '''
    Class for declaring the embedding network that creates the UMAP embedding. This class is called either from the 
    Parametric UMAP class, or from the RegularizedAutoEncoderGenerator directly if the inference parameter is set to True.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary built from the configuration .json file.

    input_shape: int
        Input parameter shape.

    output_shape: int
        Embedding dimensionality (which is set to match the latent space dimensionality of the autoencoder).
    -----------------

    Declared network is stored in the .embedder method.
    '''

    def __init__(self, config, input_shape, output_shape):
        super(Embedder_test, self).__init__(config)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.build_model()

    def build_model(self):
        
        inp = keras.Input(self.input_shape)

        x = layers.Dense(512, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)
        x = layers.Dense(512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        opt = layers.Dense(self.output_shape)(x)

        self.embedder = keras.Model(inp, opt)

class UMAPMapper_test(BaseModel):

    '''
    Regularized autoencoder mapper from the UMAP embedding to the autoencoder's latent space. Built entirely from 
    .json configuration file. Declares a the model using Keras' functional api with the appropriate input and output
    dimensions. If model.mapper_architecture field from configuration .json file is set to "dense", it defines a dense-based
    architecture with concatenations of the input from the UMAP embedding. If, on the contrary, it is set to "convolutional" it
    makes use of the ConvBlock1D class to build a residual network.

    Input arguments:
    ----------------
    config: dict 
        Configuration dictionary generated from loading the configuration .json file.
    ----------------
    Declared model is in the .mapper method.
    '''

    def __init__(self, config):
        super(UMAPMapper_test, self).__init__(config)

        if self.config.model.mapper_architecture == "dense":
            self.build_model_dense()
        elif self.config.model.mapper_architecture == "convolutional":
            self.build_model_convolutional()
        else:
            raise NameError("Mapper architecture not implemented!")

    def build_model_dense(self):

        input_output_shape = self.config.model.latent_dim

        inp = keras.Input(input_output_shape)

        x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(inp)
        x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        x = layers.Concatenate()([x, inp])
        x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        pre_opt = layers.Dense(input_output_shape)(x)

        opt = layers.Add()([inp, pre_opt])

        self.mapper = keras.Model(inp, opt)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)

        self.mapper.compile(optimizer = optimizer,
                           loss = 'mae'
                           )

    def build_model_convolutional(self):
        
        input_output_shape = self.config.model.latent_dim

        inp = keras.Input(input_output_shape)

        y1 = ConvBlock1D_test(self.config, input_output_shape).conv_block(inp)
        x1 = layers.Add()([inp, y1])

        y2 = ConvBlock1D_test(self.config, input_output_shape).conv_block(x1)
        x2 = layers.Add()([inp, y1, y2])

        y3 = ConvBlock1D_test(self.config, input_output_shape).conv_block(x2)
        opt = layers.Add()([inp, y1, y2, y3])

        self.mapper = keras.Model(inp, opt)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)

        self.mapper.compile(optimizer = optimizer,
                           loss = 'mae'
                           )
        
class MLP_test(BaseModel):

    '''
    Class for declaring the embedding network that creates the UMAP embedding. This class is called either from the 
    Parametric UMAP class, or from the RegularizedAutoEncoderGenerator directly if the inference parameter is set to True.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary built from the configuration .json file.

    input_shape: int
        Input parameter shape.

    output_shape: int
        Embedding dimensionality (which is set to match the latent space dimensionality of the autoencoder).
    -----------------

    Declared network is stored in the .embedder method.
    '''

    def __init__(self, config, input_shape, output_shape, model_id):
        super(MLP_test, self).__init__(config)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_id = str(model_id)

        self.build_model()

    def build_model(self):
        
        inp = keras.Input(self.input_shape)

        x = get_pca_model_from_id(input=inp, model_id = self.model_id)

        # if self.model_id == '0':

        #     x = layers.Dense(1024, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)
        #     x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        # elif self.model_id == '1':

        #     print("\n\nCheckpoint -1\n\n")

        #     x = layers.Dense(1024, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)
        #     x = layers.Dense(10, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        # elif self.model_id == '2':

        #     x = layers.Dense(1024, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)
        #     x = layers.Dense(10, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        # elif self.model_id == '3':

        #     x = layers.Dense(1024, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)
        #     x = layers.Dense(10, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        # elif self.model_id == '4':

        #     x = layers.Dense(1024, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)
        #     x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(10, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        # elif self.model_id == '5':

        #     x = layers.Dense(1024, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)
        #     x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(10, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        # elif self.model_id == '6':

        #     x = layers.Dense(512, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        #     x_1 = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(x_1)
        #     x = layers.Concatenate()([x, x_1])
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        #     x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        #     pre_opt = layers.Dense(512)(x)

        #     x = layers.Add()([x_1, pre_opt])

        #     y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        #     y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)
        #     x = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)

        # elif self.model_id == '7':

        #     params = keras.Input(self.in_out_shapes['input_shape'])

        #     x = layers.Dense(128, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(params) 
        #     x = layers.Dense(128, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)
        #     x = layers.Dense(128, activation = 'leaky_relu', kernel_initializer = 'glorot_uniform')(x)

        opt = layers.Dense(self.output_shape)(x)

        self.model = keras.Model(inp, opt)

class AutoEncoder_test(BaseModel):

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
        super(AutoEncoder_test, self).__init__(config)
        self.latent_dim = config.model.latent_dim
        self.out_shape = out_shape

        if self.config.data_loader.data_output_type == 'amplitude_phase':
            self.overlap = overlap_amp_phs
            self.ovlp_mae_loss = ovlp_mae_loss_amp_phs

        elif self.config.data_loader.data_output_type == 'hphc':
            self.overlap = overlap_hphc
            self.ovlp_mae_loss = ovlp_mae_loss_hphc
        else:
            self.overlap = 'mean_squared_error'

        self.build_model()

    def build_model(self):

        inp = keras.Input(self.out_shape)
        
        enc, dec = get_ae_model_from_id(input = inp, model_id = self.config.model.ae_id, latent_dimension = self.latent_dim, output_dimension = self.out_shape, config = self.config)

        self.autoencoder = keras.Model(inp, dec)
        self.encoder = keras.Model(inp, enc)
        self.decoder = keras.Model(enc, dec)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)

        if self.config.model.loss == 'overlap':

            self.autoencoder.compile(optimizer = optimizer, loss = self.ovlp_mae_loss, metrics = [self.overlap, 'mean_absolute_error'])
        else:

            self.autoencoder.compile(optimizer = optimizer, loss = 'mae', metrics = [self.overlap, 'mean_absolute_error'])

