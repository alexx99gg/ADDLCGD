import tensorflow as tf


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
    model.add(tf.keras.layers.Flatten(input_shape=(n_SNPs, 1)))

    # Dense with neurons equal to the number of inputs SNPs, ReLU as activation, L2 Regularization and He Initialization
    model.add(tf.keras.layers.Dense(1234))

    # TODO

    # OUTPUT
    # Dense with 2 outputs, sigmoid activation
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    # Compile the Neural Network
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
