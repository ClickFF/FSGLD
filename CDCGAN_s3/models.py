import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

def build_generator(latent_dim=100, n_classes=2, umask=None, mask=None):
    inputs = Input(shape=[latent_dim + n_classes])
    net = Dense(21*21, activation="relu")(inputs)
    net = Reshape((21, 21))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.2)(net) 
    net = UpSampling1D()(net)

    net = Conv1D(64, 3, activation='relu', padding='same')(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.2)(net)    
    net = UpSampling1D()(net)
    net = Dropout(0.25)(net)

    net = Conv1D(64, 3, activation='relu', padding='same')(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.2)(net)
    net = UpSampling1D()(net)
    net = Dropout(0.25)(net)

    net = Conv1D(64, 3, activation='relu', padding='same')(net)
    net = BatchNormalization()(net)
    net = LeakyReLU(alpha=0.2)(net)
    net = Dropout(0.25)(net)
    
    outputs = Conv1D(1, 3, activation='sigmoid', padding='same')(net)

    def custom_layer(x):
        return x * umask + mask

    lambda_layer = Lambda(custom_layer)(outputs)
    return Model(inputs, lambda_layer)

def build_discriminator(input_dim=168, n_classes=2):
    inputs = Input(shape=[input_dim, 1 + n_classes])
    net = Conv1D(64, 11, strides=4, activation='relu', 
                kernel_regularizer=l2(0.01), padding='valid')(inputs)
    net = MaxPooling1D(3, strides=2)(net)
    
    net = Conv1D(128, 5, activation='relu', strides=1, 
                kernel_regularizer=l2(0.01), padding='same')(net)
    net = MaxPooling1D(3, strides=2)(net)
    
    net = Conv1D(256, 3, activation='relu', strides=1, 
                kernel_regularizer=l2(0.01), padding='same')(net)
    net = Conv1D(256, 3, activation='relu', strides=1, 
                kernel_regularizer=l2(0.01), padding='same')(net)
    net = Conv1D(128, 3, activation='relu', strides=1, 
                kernel_regularizer=l2(0.01), padding='same')(net)
    net = Dropout(0.5)(net)
    net = Flatten()(net)
    
    net = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(net)
    net = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(net)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(net)
    return Model(inputs, outputs)

def to_categorical(y, depth=2):
    y = tf.cast(y, tf.int32) 
    if len(y.shape) == 0 or (len(y.shape) == 1 and y.shape[0] is None):
        y = tf.reshape(y, [1])  
    if len(y.shape) > 1 and y.shape[-1] == 1:
        y = tf.squeeze(y, axis=-1)  
    return tf.one_hot(y, depth=depth)  

def concatenate_data_label(data, labels, fingerprint_dim=168):
    labels_one_hot = to_categorical(labels)  # (N, 2)
    labels_tiled = tf.tile(labels_one_hot[:, tf.newaxis, :], [1, fingerprint_dim, 1])  # (N,168,2)
    data = tf.cast(data, tf.float32)  
    return tf.concat([data, labels_tiled], axis=-1)  # (N,168,3)
    