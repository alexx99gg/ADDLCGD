import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import layers


def create_model(n_SNPs: int):
    # Define Neural Network architecture
    """
    Extracted from the original paper (https://www.biorxiv.org/content/10.1101/629402v1.full.pdf):
    • Input: SNPs obtained from QualityControl Pipeline
    • Dense with neurons equal to the number of inputs SNPs, ReLU as activation, L2 Regularization and He Initialization
    • Batch Normalization, Dropout Layer with 30% of inputs to drop,Gaussian Noise with 0.3 as Standard Deviation

    • Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    • Batch Normalization, Dropout Layer with 30% of inputs to drop
    • Dense Layer with 512 outputs, ReLU as activation, L2 Regularization and He Initialization
    • Batch Normalization, Dropout Layer with 30% of inputs to drop
    • Dense Layer with 256 outputs, ReLU as activation, L2 Regularization and He Initialization
    • Batch Normalization, Dropout Layer with 30% of inputs to drop
    • Dense Layer with 64 outputs, ReLU as activation, L2 Regularization and He Initialization
    • Dense with 2 outputs, sigmoid activation
    • Output: Prediction probability for Alzheimer’s Disease
    """
    model = tf.keras.models.Sequential()
    model.add(layers.Flatten(input_shape=(n_SNPs, 1)))

    # Dense with neurons equal to the number of inputs SNPs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 14, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop,Gaussian Noise with 0.3 as Standard Deviation
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.GaussianNoise(0.3))

    # Dense with neurons equal to the number of inputs SNPs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 13, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop,Gaussian Noise with 0.3 as Standard Deviation
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense with neurons equal to the number of inputs SNPs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 12, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop,Gaussian Noise with 0.3 as Standard Deviation
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 11, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 10, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 9, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 8, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 7, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 6, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # Dense Layer with 1024 outputs, ReLU as activation, L2 Regularization and He Initialization
    model.add(
        layers.Dense(2 ^ 5, activation='relu', kernel_regularizer='l2', kernel_initializer=initializers.he_normal))
    # Batch Normalization, Dropout Layer with 30% of inputs to drop
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # OUTPUT
    # Dense with 2 outputs, sigmoid activation
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the Neural Network
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    return model
