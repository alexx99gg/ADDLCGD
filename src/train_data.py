import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import keras.backend as K


def f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def create_MLP_model(n_SNPs: int):
    # Define Neural Network architecture

    model = tf.keras.models.Sequential()
    model.add(layers.Flatten(input_shape=(n_SNPs, 1)))

    # Dense with neurons equal to the number of inputs SNPs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(n_SNPs, activation='elu', kernel_regularizer='l1',
                     kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop,Gaussian Noise with 0.2 as Standard Deviation
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    # model.add(layers.GaussianNoise(0.2))

    # Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(1024, activation='elu', kernel_regularizer='l1', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 512 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(512, activation='elu', kernel_regularizer='l1', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 256 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(256, activation='elu', kernel_regularizer='l1', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 64 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(64, activation='elu', kernel_regularizer='l1', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # OUTPUT
    # Dense with 2 outputs, sigmoid activation
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the Neural Network
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    return model
