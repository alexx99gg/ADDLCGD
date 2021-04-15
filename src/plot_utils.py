import matplotlib.pyplot as plt
import numpy as np

def plot_2d_dataset(x, y):
  x_alzheimer = []
  x_no_alzheimer = []
  for i in range(len(y)):
    if y[i] == 1:
        x_alzheimer.append(x[i])
    elif y[i] == 0:
      x_no_alzheimer.append(x[i])
  x_alzheimer = np.asarray(x_alzheimer)
  x_no_alzheimer = np.asarray(x_no_alzheimer)

  plt.plot(x_no_alzheimer[:,0], x_no_alzheimer[:,1], 'b.', label='No alzheimer')
  plt.plot(x_alzheimer[:,0], x_alzheimer[:,1], 'r.', label='Alzheimer')
  plt.legend()
  plt.show()



def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
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
