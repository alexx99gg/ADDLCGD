import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score

plt.rcParams['font.size'] = '16'


def plot_2d_dataset(x_2d, y, xmin, xmax, ymin, ymax, source: str):
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
    plt.title(f"{source} 2D data representation")
    plt.grid()
    plt.savefig(f"../results/{source}_2d_data.png")
    plt.show()


def plot_training_history(history):
    # Accuracy history
    plt.plot(history.history['accuracy'], label='Train', color='blue', alpha=0.7, linewidth=3)
    plt.plot(history.history['val_accuracy'], label='Validation', color='red', alpha=0.7, linewidth=3)
    plt.legend()
    plt.title('Training accuracy history')
    plt.ylim(ymin=0, ymax=1)
    plt.grid()
    plt.savefig("../results/training_accuracy_history.png")
    plt.show()

    # AUC history
    plt.plot(history.history['auc'], label='Train', color='blue', alpha=0.7, linewidth=3)
    plt.plot(history.history['val_auc'], label='Validation', color='red', alpha=0.7, linewidth=3)
    plt.legend()
    plt.title('Training AUC history')
    plt.ylim(ymin=0, ymax=1)
    plt.grid()
    plt.savefig("../results/training_auc_history.png")
    plt.show()


def plot_confusion_matrix(y, y_prob, y_pred, model: str):
    cm = metrics.confusion_matrix(y, y_pred)
    df_cm = pd.DataFrame(cm, index=["CN", "AD"], columns=["CN", "AD"])

    sn.heatmap(df_cm, vmin=0, vmax=np.max(np.sum(cm, axis=1)), annot=True, cmap='Purples', fmt='d')

    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")

    plt.title(f"{model} confusion matrix")
    plt.savefig(f"../results/{model}_confusion_matrix.png")
    plt.show()

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    print(f"{model} Precision {precision:.2f} Recall {recall:.2f} AUC {auc:.2f}")


def plot_roc_curve(DNN_auc_score_list, DNN_tpr_matrix, SVM_auc_score_list, SVM_tpr_matrix,
               RF_auc_score_list, RF_tpr_matrix, GB_auc_score_list, GB_tpr_matrix):
    # Draw diagonal reference line
    line_x_y = np.linspace(0, 1, 100)
    plt.plot(line_x_y, line_x_y, color='red', alpha=0.7, linewidth=2.25, linestyle='dashed')

    DNN_mean_auc_score = np.mean(DNN_auc_score_list)
    SVM_mean_auc_score = np.mean(SVM_auc_score_list)
    RF_mean_auc_score = np.mean(RF_auc_score_list)
    GB_mean_auc_score = np.mean(GB_auc_score_list)

    base_fpr = np.linspace(0, 1, 101)

    DNN_mean_tpr = DNN_tpr_matrix.mean(axis=0)
    SVM_mean_tpr = SVM_tpr_matrix.mean(axis=0)
    RF_mean_tpr = RF_tpr_matrix.mean(axis=0)
    GB_mean_tpr = GB_tpr_matrix.mean(axis=0)



    # Draw roc curve
    plt.plot(base_fpr, DNN_mean_tpr, linewidth=6, color='blue', alpha=0.7, label=f"DNN (mean AUC {DNN_mean_auc_score:.2f})")
    plt.plot(base_fpr, SVM_mean_tpr, linewidth=5, color='purple', alpha=0.7, label=f"SVM (mean AUC {SVM_mean_auc_score:.2f})")
    plt.plot(base_fpr, RF_mean_tpr, linewidth=4, color='orange', alpha=0.7, label=f"RF (mean AUC {RF_mean_auc_score:.2f})")
    plt.plot(base_fpr, GB_mean_tpr, linewidth=3, color='green', alpha=0.7, label=f"GB (mean AUC {GB_mean_auc_score:.2f})")

    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC curve')
    plt.grid()
    plt.savefig("../results/roc_curve.png")
    plt.show()
