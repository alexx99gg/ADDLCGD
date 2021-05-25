import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.decomposition import PCA


def plot_2d_dataset(x_2d, y):
    x_alzheimer = []
    x_no_alzheimer = []
    for i in range(len(y)):
        if y[i] == 1:
            x_alzheimer.append(x_2d[i])
        elif y[i] == 0:
            x_no_alzheimer.append(x_2d[i])
    x_alzheimer = np.asarray(x_alzheimer)
    x_no_alzheimer = np.asarray(x_no_alzheimer)

    plt.plot(x_no_alzheimer[:, 0], x_no_alzheimer[:, 1], 'b.', label='No alzheimer', alpha=0.33)
    plt.plot(x_alzheimer[:, 0], x_alzheimer[:, 1], 'r.', label='Alzheimer', alpha=0.33)
    plt.legend()
    plt.title('2D data representation')
    plt.show()


def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Train', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation', color='red')
    plt.legend()
    plt.title('Training accuracy history')
    plt.ylim(ymin=0, ymax=1)
    plt.show()

    plt.plot(history.history['loss'], label='Train', color='blue')
    plt.plot(history.history['val_loss'], label='Validation', color='red')
    plt.legend()
    plt.title('Training loss history')
    plt.ylim(ymin=0)
    plt.show()


def plot_confusion_matrix(y, y_prob):
    y_pred = np.rint(y_prob)
    cm = metrics.confusion_matrix(y, y_pred)
    df_cm = pd.DataFrame(cm, index=["CN", "AD"], columns=["CN", "AD"])

    sn.heatmap(df_cm, vmin=0, annot=True, cmap='Purples', fmt='d')

    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")

    plt.title('Confusion matrix')
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

    plt.title('ROC curve')
    plt.show()
