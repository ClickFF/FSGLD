from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

def build_generator(latent_dim=100):
    model = Sequential([
        Dense(21*21, activation="relu", input_dim=latent_dim),
        Reshape((21, 21)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        UpSampling1D(),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        UpSampling1D(),
        Dropout(0.25),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        UpSampling1D(),
        Dropout(0.25),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv1D(1, 3, activation='sigmoid', padding='same')
    ])
    return model

def build_discriminator(input_dim=168):
    inputs = Input(shape=(input_dim, 1))
    x = Conv1D(64, 11, strides=4, activation='relu', 
               kernel_regularizer=l2(0.01), padding='valid')(inputs)
    x = MaxPooling1D(3, strides=2)(x)
    x = Conv1D(128, 5, activation='relu', strides=1, 
               kernel_regularizer=l2(0.01), padding='same')(x)
    x = MaxPooling1D(3, strides=2)(x)
    x = Conv1D(256, 3, activation='relu', strides=1, 
               kernel_regularizer=l2(0.01), padding='same')(x)
    x = Conv1D(256, 3, activation='relu', strides=1, 
               kernel_regularizer=l2(0.01), padding='same')(x)
    x = Conv1D(128, 3, activation='relu', strides=1, 
               kernel_regularizer=l2(0.01), padding='same')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)
    return Model(inputs, outputs)