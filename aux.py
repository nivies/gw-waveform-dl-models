import os
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from utils.eval import load_model_and_data_loader
from data_loader.gw_dataloader import DataGenerator

'Get autoencoder architecture back to normal for loading'

cpu = False

if cpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=''
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


model_dir = "/home/nino/GW/Keras-Project-Template/experiments/reg_ae_osv_ok/reg_ae_osv_bbh/checkpoints"
scaler = StandardScaler()
bs = 128
epochs = 1000


callbacks = []
callbacks.append(EarlyStopping(monitor = 'val_loss', patience = 30))
callbacks.append(ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 10))

model, data_loader = load_model_and_data_loader(model_dir)

data_loader.X_train = scaler.fit_transform(data_loader.X_train)
data_loader.X_test = scaler.transform(data_loader.X_test)


optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss = 'mae', optimizer = optimizer)

retrain_history = model.fit(
    x = DataGenerator(data_loader.X_train, data_loader.y_train, bs),
    validation_data = DataGenerator(data_loader.X_test, data_loader.y_test, bs),
    epochs = epochs,
    callbacks = callbacks
)

model.save_weights(os.path.join(model_dir, "best_model_retrained.h5"))