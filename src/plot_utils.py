import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score


def plot_2d_dataset(x_2d, y, xmin, xmax, ymin, ymax):
    x_alzheimer = []
    x_no_alzheimer = []
    for i in range(len(y)):
        if y[i] == 1:
            x_alzheimer.append(x_2d[i])
        elif y[i] == 0:
            x_no_alzheimer.append(x_2d[i])
    x_alzheimer = np.asarray(x_alzheimer)
    x_no_alzheimer = np.asarray(x_no_alzheimer)

    plt.plot(x_no_alzheimer[:, 0], x_no_alzheimer[:, 1], 'bo', label='No alzheimer', alpha=0.33)
    plt.plot(x_alzheimer[:, 0], x_alzheimer[:, 1], 'ro', label='Alzheimer', alpha=0.33)

    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.legend()
    plt.title('2D data representation')
    plt.grid()
    plt.show()


def plot_training_history(history):
    # Accuracy history
    plt.plot(history.history['accuracy'], label='Train', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation', color='red')
    plt.legend()
    plt.title('Training accuracy history')
    plt.ylim(ymin=0, ymax=1)
    plt.grid()
    plt.show()

    # AUC history
    plt.plot(history.history['auc'], label='Train', color='blue')
    plt.plot(history.history['val_auc'], label='Validation', color='red')
    plt.legend()
    plt.title('Training AUC history')
    plt.ylim(ymin=0, ymax=1)
    plt.grid()
    plt.show()


def plot_confusion_matrix(y, y_prob, model: str):
    y_pred = np.rint(y_prob)
    cm = metrics.confusion_matrix(y, y_pred)
    df_cm = pd.DataFrame(cm, index=["CN", "AD"], columns=["CN", "AD"])

    sn.heatmap(df_cm, vmin=0, vmax=np.max(np.sum(cm, axis=1)), annot=True, cmap='Purples', fmt='d')

    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")

    plt.title(f"{model} confusion matrix")
    plt.show()

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    print(f"{model} Precision {precision:.2f} Recall {recall:.2f} AUC {auc:.2f}")


def plot_roc_curve(y, DNN_y_prob, SVC_y_prob, RF_y_prob, GBC_y_prob):

    DNN_auc_score = roc_auc_score(y, DNN_y_prob)
    DNN_fpr, DNN_tpr, _ = roc_curve(y, DNN_y_prob)
    
    SVC_auc_score = roc_auc_score(y, SVC_y_prob)
    SVC_fpr, SVC_tpr, _ = roc_curve(y, SVC_y_prob)
    
    RF_auc_score = roc_auc_score(y, RF_y_prob)
    RF_fpr, RF_tpr, _ = roc_curve(y, RF_y_prob)
    
    GBC_auc_score = roc_auc_score(y, GBC_y_prob)
    GBC_fpr, GBC_tpr, _ = roc_curve(y, GBC_y_prob)

    # Draw roc curve
    plt.plot(DNN_fpr, DNN_tpr, linewidth=4, marker='o', color='blue', label=f"DNN (AUC {DNN_auc_score:.2f})")
    plt.plot(SVC_fpr, SVC_tpr, linewidth=4, marker='o', color='purple', label=f"SVC (AUC {SVC_auc_score:.2f})")
    plt.plot(RF_fpr, RF_tpr, linewidth=4, marker='o', color='orange', label=f"RF (AUC {RF_auc_score:.2f})")
    plt.plot(GBC_fpr, GBC_tpr, linewidth=4, marker='o', color='green', label=f"GBC (AUC {GBC_auc_score:.2f})")

    # Draw diagonal reference line
    x = np.linspace(0, 1, 100)
    y = x
    plt.plot(x, y, color='red', linewidth=2.25, linestyle='dashed')

    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC curve')
    plt.grid()
    plt.show()
