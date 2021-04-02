import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Draw straight line
    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, '--r')
    plt.show()
