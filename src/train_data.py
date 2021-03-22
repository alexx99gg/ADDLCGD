import tensorflow as tf

# Extract Single Nucleotide Polymorphisms (SNPs)
# TODO

# number of single nucleotide polymorphisms
n_SNPs = 1234

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
model = tf.keras.models.Sequential
model.add(tf.keras.layers.Flatten(input_shape=n_SNPs))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2,
                                kernel_initializer=tf.keras.initializers.he_normal))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

# Compile the Neural Network
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

