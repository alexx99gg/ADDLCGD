import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import shap
from qqman import qqman
from sklearn.metrics import auc

import settings

plt.rcParams['font.size'] = '14'


def plot_2d_dataset(x_2d, y, xmin, xmax, ymin, ymax, source: str, fold: int):
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
    plt.title(f"{source} 2D data representation for fold {fold}")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{settings.save_dir}{source}_2d_data_fold_{fold}.png")
    plt.show()


def plot_training_history(history, fold: int):
    # Accuracy history
    plt.plot(history.history['accuracy'], label='Train', color='blue', alpha=0.7, linewidth=3)
    plt.plot(history.history['val_accuracy'], label='Validation', color='red', alpha=0.7, linewidth=3)
    plt.legend()
    plt.title(f"Training accuracy history for fold {fold}")
    plt.ylim(ymin=0, ymax=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{settings.save_dir}training_accuracy_history_fold_{fold}.png")
    plt.show()

    # AUC history
    plt.plot(history.history['auc'], label='Train', color='blue', alpha=0.7, linewidth=3)
    plt.plot(history.history['val_auc'], label='Validation', color='red', alpha=0.7, linewidth=3)
    plt.legend()
    plt.title(f"Training AUC history for fold {fold}")
    plt.ylim(ymin=0, ymax=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{settings.save_dir}training_auc_history_fold_{fold}.png")
    plt.show()


def plot_confusion_matrix(cm, model_name: str):
    df_cm = pd.DataFrame(cm, index=["CN", "AD"], columns=["CN", "AD"])

    sn.heatmap(df_cm, vmin=0, vmax=np.max(np.sum(cm, axis=1)), annot=True, cmap='Purples', fmt='d')

    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"{model_name} confusion matrix")

    plt.tight_layout()
    plt.savefig(f"{settings.save_dir}{model_name}_confusion_matrix_fold.png")
    plt.show()


def plot_roc_curve(DNN_auc_score_list, DNN_tpr_matrix, SVM_auc_score_list, SVM_tpr_matrix,
                   RF_auc_score_list, RF_tpr_matrix, GB_auc_score_list, GB_tpr_matrix):
    base_fpr = np.linspace(0, 1, 101)

    DNN_mean_tpr = np.mean(DNN_tpr_matrix, axis=0)
    SVM_mean_tpr = np.mean(SVM_tpr_matrix, axis=0)
    RF_mean_tpr = np.mean(RF_tpr_matrix, axis=0)
    GB_mean_tpr = np.mean(GB_tpr_matrix, axis=0)

    DNN_mean_auc_score = auc(base_fpr, DNN_mean_tpr)
    SVM_mean_auc_score = auc(base_fpr, SVM_mean_tpr)
    RF_mean_auc_score = auc(base_fpr, RF_mean_tpr)
    GB_mean_auc_score = auc(base_fpr, GB_mean_tpr)

    DNN_std_auc = np.std(DNN_auc_score_list, axis=0)
    SVM_std_auc = np.std(SVM_auc_score_list, axis=0)
    RF_std_auc = np.std(RF_auc_score_list, axis=0)
    GB_std_auc = np.std(GB_auc_score_list, axis=0)

    # Draw roc curve
    plt.plot(base_fpr, DNN_mean_tpr, linewidth=6, color='blue', alpha=0.7,
             label=f"DNN (mean AUC {DNN_mean_auc_score:.2f} $\pm$ {DNN_std_auc:.2f})")
    plt.plot(base_fpr, SVM_mean_tpr, linewidth=5, color='purple', alpha=0.7,
             label=f"SVM (mean AUC {SVM_mean_auc_score:.2f} $\pm$ {SVM_std_auc:.2f})")
    plt.plot(base_fpr, RF_mean_tpr, linewidth=4, color='orange', alpha=0.7,
             label=f"RF (mean AUC {RF_mean_auc_score:.2f} $\pm$ {RF_std_auc:.2f})")
    plt.plot(base_fpr, GB_mean_tpr, linewidth=3, color='green', alpha=0.7,
             label=f"GB (mean AUC {GB_mean_auc_score:.2f} $\pm$ {GB_std_auc:.2f})")

    # Draw diagonal reference line
    line_x_y = np.linspace(0, 1, 100)
    plt.plot(line_x_y, line_x_y, color='red', alpha=0.7, linewidth=2.25, linestyle='dashed',
             label="Chance")

    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('Mean ROC curves')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{settings.save_dir}roc_curve.png")
    plt.show()


def plot_snp(selected_snp_names, selected_snp_p_values, fold: int):
    fig, ax = plt.subplots()
    ax.barh(selected_snp_names[:20], selected_snp_p_values[:20], color='blue', height=0.7)
    ax.invert_yaxis()
    plt.grid(axis='x')
    ax.set_xscale('log')
    ax.set_xlabel('P-value')
    ax.set_ylabel('SNP name')
    ax.set_title(f"SNPs p-values in GWAS for fold {fold}")
    plt.tight_layout()
    plt.savefig(f"{settings.save_dir}snps_fold_{fold}.png")
    plt.show()


def plot_manhattan(assoc_path, fold):
    qqman.manhattan(assoc_path, title=f"Manhattan plot of fold {fold}", show=True,
                    out=f"{settings.save_dir}manhattan_fold_{fold}.png", cmap=plt.get_cmap('turbo'),
                    cmap_var=7)


def plot_shap(model, X_train, X_test, y_train, y_test, fold: int, model_name: str):
    xmin = -0.75
    xmax = 0.75

    X_train = X_train.iloc[:50, :]

    explainer = shap.KernelExplainer(model=model, data=X_train, link="identity")
    shap_values = explainer.shap_values(X=X_test, nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    try:

        shap.summary_plot(shap_values=shap_values, features=X_test, show=False, plot_type="violin")
        plt.xlim(xmin=xmin, xmax=xmax)
        plt.title(f"SNPs SHAP values in {model_name} for fold {fold}")
        plt.tight_layout()
        plt.savefig(f"{settings.save_dir}feature_shap_{model_name}_fold_{fold}.png")
        plt.show()
    except np.linalg.LinAlgError:
        print(f"WARNING: can't calculate SHAP values in {model_name} for fold {fold}")
        pass

