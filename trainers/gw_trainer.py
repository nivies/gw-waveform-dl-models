from base.base_trainer import BaseTrain
import os
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from joblib import dump

'''
File for declaring the training classes. Every class inherits from the BaseTrain class defined in the base directory.
'''

class GWModelTrainer(BaseTrain):

    '''
    Class for the training of the dense and cVAE GW generation models. Defines the ModelCheckpoint, Tensorboard, EarlyStopping and ReduceLROnPlateau callbacks
    and calls the .fit method for an already compiled model. 

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary built from the configuration .json file.
    
    model: Instance from MLP class.
        Model tasked with the GW generation.

    data: data_loader class instance
        data_loader instance called for the particular problem. All information for this class' calling is contained in the configuration file.
    -----------------

    The ModelCheckpoint automatically saves the model. The training history is also saved to the same directory, serialized using the joblib library.
    '''

    def __init__(self, config, model, data):
        super(GWModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.model = model.model
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                # filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, 'best_model.hdf5'),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )
        self.callbacks.append(EarlyStopping(monitor = 'val_loss', patience = self.config.callbacks.early_stopping_patience, min_delta = 1e-6))
        self.callbacks.append(ReduceLROnPlateau(monitor = 'loss', factor = self.config.callbacks.lr_reduce_factor, patience = self.config.callbacks.lr_reduce_patience, verbose = 1, min_lr = self.config.callbacks.min_lr))

    def train(self):

        history = self.model.fit(
            self.data.get_train_data(),
            validation_data = self.data.get_test_data(),
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks
        )
        self.loss.extend(history.history['loss'])
        self.val_loss.extend(history.history['val_loss'])
        self.history = history.history
        dump(history, os.path.join(self.config.callbacks.checkpoint_dir, "history.bin"))
        
class GWRegularizedAutoEncoderModelTrainer(BaseTrain):

    '''
    Class for the training of the regularized autoencoder based GW generation models. Defines the ModelCheckpoint, Tensorboard, 
    EarlyStopping and ReduceLROnPlateau callbacks and calls the .fit method for an already compiled model for every needed model 
    and the final retraining. 

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary built from the configuration .json file.
    
    model: Instance from MLP class.
        Model tasked with the GW generation.

    data: data_loader class instance
        data_loader instance called for the particular problem. All information for this class' calling is contained in the configuration file.
    -----------------

    The ModelCheckpoint automatically saves the best model, along with every sub network best weights. The training histories are also saved to 
    the same directory, serialized using the joblib library.
    '''

    def __init__(self, config, model, data):
        super(GWRegularizedAutoEncoderModelTrainer, self).__init__(model, data, config)

        self.mapper = model.mapper
        self.autoencoder = model.autoencoder
        self.embedder = model.embedder
        self.generator = model
        self.embedder_epochs = config.parametric_umap.epochs
        self.callbacks_common = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
    
        checkpoint_ae = ModelCheckpoint(
                # filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, 'best_autoencoder.hdf5'),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        
        checkpoint_mapper = ModelCheckpoint(
                # filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, 'best_mapper.hdf5'),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        
        checkpoint_generator = ModelCheckpoint(
                # filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, 'best_model.hdf5'),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )

        self.callbacks_common.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )
        self.callbacks_common.append(EarlyStopping(monitor = 'val_loss', patience = self.config.callbacks.early_stopping_patience, min_delta = 1e-6))
        self.callbacks_common.append(ReduceLROnPlateau(monitor = 'loss', factor = self.config.callbacks.lr_reduce_factor, patience = self.config.callbacks.lr_reduce_patience, verbose = 1, min_lr = self.config.callbacks.min_lr))

        self.callbacks_embedder = self.callbacks_common
        self.callbacks_ae = [checkpoint_ae] + self.callbacks_common
        self.callbacks_mapper = [checkpoint_mapper] + self.callbacks_common
        self.callbacks_generator = [checkpoint_generator] + self.callbacks_common

    def train(self):
        
        print("\n\nParametric UMAP training\n\n")

        # create embedding
        history_embedder = self.embedder.parametric_model.fit(
            self.embedder.edge_dataset,
            epochs = self.embedder_epochs,
            steps_per_epoch = self.embedder.steps_per_epoch,
            callbacks = self.callbacks_embedder,
            verbose = self.config.trainer.verbose_training
        )

        self.embedder.parametric_model.save_weights(os.path.join(self.config.callbacks.checkpoint_dir, 'best_embedder.hdf5'))
        self.data.add_umap_data(self.embedder.embedder.predict(self.data.X_train, batch_size = 1024, verbose = 0), self.embedder.embedder.predict(self.data.X_test, batch_size = 1024, verbose = 0))

        print("\n\nRegularized autoencoder training\n\n")

        history_autoencoder = self.autoencoder.autoencoder.fit(
            x = self.data.get_autoencoder_train_data(),
            validation_data = self.data.get_autoencoder_test_data(),
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks_ae
        ) 

        self.autoencoder.autoencoder.load_weights(os.path.join(self.config.callbacks.checkpoint_dir, 'best_autoencoder.hdf5'))
        self.data.add_latent_data(self.autoencoder.encoder.predict(self.data.y_train, batch_size = 1024, verbose = 0), self.autoencoder.encoder.predict(self.data.y_test, batch_size = 1024, verbose = 0))

        print("\n\nMapper training\n\n")

        history_mapper = self.mapper.fit(
            self.data.get_mapper_train_data(),
            validation_data = self.data.get_mapper_test_data(),
            epochs=self.config.trainer.mapper_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks_mapper
        )   

        print("\n\nTraining complete!\n\n")

        self.mapper.load_weights(os.path.join(self.config.callbacks.checkpoint_dir, 'best_mapper.hdf5'))
        self.generator.model.compile(optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs), loss = 'mae')

        history_retrain = self.generator.model.fit(
            self.data.get_generator_train_data(),
            validation_data = self.data.get_generator_test_data(),
            epochs=self.config.trainer.generator_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks_generator
        )

        dump(history_embedder.history, os.path.join(self.config.callbacks.checkpoint_dir, "history_embdedder.bin"))
        dump(history_autoencoder.history, os.path.join(self.config.callbacks.checkpoint_dir, "history_autoencoder.bin"))
        dump(history_mapper.history, os.path.join(self.config.callbacks.checkpoint_dir, "history_mapper.bin"))
        dump(history_retrain.history, os.path.join(self.config.callbacks.checkpoint_dir, "history_generator.bin"))

class GWMappedAutoEncoderModelTrainer(BaseTrain):

    '''
    Class for the training of the mapped autoencoder based GW generation models. Defines the ModelCheckpoint, Tensorboard, 
    EarlyStopping and ReduceLROnPlateau callbacks and calls the .fit method for an already compiled model for every needed model 
    and the final retraining. 

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary built from the configuration .json file.
    
    model: Instance from MLP class.
        Model tasked with the GW generation.

    data: data_loader class instance
        data_loader instance called for the particular problem. All information for this class' calling is contained in the configuration file.
    -----------------

    The ModelCheckpoint automatically saves the best model, along with every sub network best weights. The training histories are also saved to 
    the same directory, serialized using the joblib library.
    '''

    def __init__(self, config, model, data):
        super(GWMappedAutoEncoderModelTrainer, self).__init__(model, data, config)

        self.mapper = model.mapper
        self.autoencoder = model.autoencoder
        self.generator = model
        self.callbacks_common = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
    
        checkpoint_ae = ModelCheckpoint(
                # filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, 'best_autoencoder.hdf5'),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        
        checkpoint_mapper = ModelCheckpoint(
                # filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, 'best_mapper.hdf5'),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        
        checkpoint_generator = ModelCheckpoint(
                # filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, 'best_model.hdf5'),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )


        self.callbacks_common.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )
        self.callbacks_common.append(EarlyStopping(monitor = 'val_loss', patience = self.config.callbacks.early_stopping_patience, min_delta = 1e-6))
        self.callbacks_common.append(ReduceLROnPlateau(monitor = 'loss', factor = self.config.callbacks.lr_reduce_factor, patience = self.config.callbacks.lr_reduce_patience, verbose = 1, min_lr = self.config.callbacks.min_lr))

        self.callbacks_embedder = self.callbacks_common
        self.callbacks_ae = [checkpoint_ae] + self.callbacks_common
        self.callbacks_mapper = [checkpoint_mapper] + self.callbacks_common
        self.callbacks_generator = [checkpoint_generator] + self.callbacks_common

    def train(self):

        print("\n\nAutoencoder training\n\n")

        history_autoencoder = self.autoencoder.autoencoder.fit(
            x = self.data.get_autoencoder_train_data(),
            validation_data = self.data.get_autoencoder_test_data(),
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks_ae
        )

        self.autoencoder.autoencoder.load_weights(os.path.join(self.config.callbacks.checkpoint_dir, 'best_autoencoder.hdf5'))
        self.data.add_latent_data(self.autoencoder.encoder.predict(self.data.y_train, batch_size = 1024, verbose = 0), self.autoencoder.encoder.predict(self.data.y_test, batch_size = 1024, verbose = 0))

        print("\n\nMapper training\n\n")

        history_mapper = self.mapper.fit(
            self.data.get_mapper_train_data(),
            validation_data = self.data.get_mapper_test_data(),
            epochs=self.config.trainer.mapper_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks_mapper
        )   

        self.mapper.load_weights(os.path.join(self.config.callbacks.checkpoint_dir, 'best_mapper.hdf5'))
        self.generator.model.compile(optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs), loss = 'mae')

        print("\n\nRetraining generator...\n\n")

        history_retrain = self.generator.model.fit(
            self.data.get_generator_train_data(),
            validation_data = self.data.get_generator_test_data(),
            epochs=self.config.trainer.generator_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks_generator
        )

        print("\n\nTraining complete!\n\n")

        dump(history_autoencoder.history, os.path.join(self.config.callbacks.checkpoint_dir, "history_autoencoder.bin"))
        dump(history_mapper.history, os.path.join(self.config.callbacks.checkpoint_dir, "history_mapper.bin"))
        dump(history_retrain.history, os.path.join(self.config.callbacks.checkpoint_dir, "history_generator.bin"))