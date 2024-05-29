import tensorflow as tf
import keras.backend as K
from keras.losses import mean_absolute_error

def overlap(h1, h2, dt=2*4.925794970773135e-06, df=None):
    
    split_size = int(4096/2)

    h1_amp, h1_phs = h1[:, :split_size], h1[:, split_size:]
    h2_amp, h2_phs = h2[:, :split_size], h2[:, split_size:]

    h1 =  tf.cast(h1_amp, tf.complex64)*tf.math.exp(1j*(tf.cast(h1_phs, tf.complex64)))
    h2 = tf.cast(h2_amp, tf.complex64)*tf.math.exp(1j*(tf.cast(h2_phs, tf.complex64)))

    h1_f = tf.signal.fft(h1)*dt
    h2_f = tf.signal.fft(h2)*dt
    
    df = 1.0 /  2048 / dt
    sig_norm = 4*df

    sig1 = K.sqrt(tf.cast((tf.math.reduce_sum(tf.math.conj(h1_f)*h1_f,axis=-1)),tf.float32)*sig_norm)
    sig2 = K.sqrt(tf.cast((tf.math.reduce_sum(tf.math.conj(h2_f)*h2_f,axis=-1)),tf.float32)*sig_norm)
    
    norm = 1/sig1/sig2
    inner = tf.cast(tf.math.reduce_sum((tf.math.conj(h1_f)*h2_f),axis=-1),tf.float32)
    overl = tf.cast((4*df*inner*norm),tf.float32)
    
    return  K.abs(1. - overl)

def ovlp_mae_loss(y_pred, y_true):
    
    return overlap(y_pred, y_true) + mean_absolute_error(y_pred, y_true)