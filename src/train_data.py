import keras.backend as K
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import settings


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
        layers.Dense(n_SNPs, activation='relu', kernel_regularizer='l1_l2',
                     kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop,Gaussian Noise with 0.2 as Standard Deviation
    model.add(layers.GaussianNoise(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(1024, activation='relu', kernel_regularizer='l1_l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 512 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(512, activation='relu', kernel_regularizer='l1_l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 256 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(256, activation='relu', kernel_regularizer='l1_l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 64 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(64, activation='relu', kernel_regularizer='l1_l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # OUTPUT
    # Dense with 2 outputs, sigmoid activation
    model.add(layers.Dense(1, activation='sigmoid'))

    tf.keras.utils.plot_model(model, to_file=f"{settings.save_dir}DNN_model_plot.png", show_shapes=True,
                              show_layer_names=True)

    # Compile the Neural Network
    model.compile(
        optimizer=optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    return model
