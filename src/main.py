import numpy as np
from pandas_plink import read_plink
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from plot_utils import *
from read_data import *
from train_data import *

dataset_path = "../wgs_data/subsets/"
dataset = "ADNI1GO23"
print(f"Reading dataset {dataset}")

DNN_cm_sum = np.zeros((2, 2), dtype=np.int)
DNN_precision_list = []
DNN_recall_list = []
DNN_auc_score_list = []
DNN_tpr_matrix = []

SVM_cm_sum = np.zeros((2, 2), dtype=np.int)
SVM_precision_list = []
SVM_recall_list = []
SVM_auc_score_list = []
SVM_tpr_matrix = []

RF_cm_sum = np.zeros((2, 2), dtype=np.int)
RF_precision_list = []
RF_recall_list = []
RF_auc_score_list = []
RF_tpr_matrix = []

GB_cm_sum = np.zeros((2, 2), dtype=np.int)
GB_precision_list = []
GB_recall_list = []
GB_auc_score_list = []
GB_tpr_matrix = []

base_fpr = np.linspace(0, 1, 101)

folds = [1, 2, 3, 4, 5]
for fold in folds:
    print()
    print(f"Fold number {fold}")
    clumped_path = f"{dataset_path}{dataset}_fold_{fold}_train.assoc.fisher"
    train_path = f"{dataset_path}{dataset}_fold_{fold}_train"
    test_path = f"{dataset_path}{dataset}_fold_{fold}_test"

    # Get SNPs to keep
    selected_snp_names, selected_snp_p_values = get_selected_snps(clumped_path)
    # Get the first ones
    selected_snp_names = selected_snp_names[:15]
    selected_snp_p_values = selected_snp_p_values[:15]

    plot_snp(selected_snp_names, selected_snp_p_values, fold)

    # Load train data
    (bim, fam, bed) = read_plink(train_path, verbose=False)
    x_train, y_train = generate_dataset(bed, bim, fam, selected_snp_names)
    x_train, y_train = shuffle(x_train, y_train)

    n_train_samples = x_train.shape[0]
    n_train_SNPs = x_train.shape[1]

    print(f"Number of SNPs in train: {n_train_SNPs}")
    n_control_train, n_case_train = count_case_control(y_train)
    print(f"Number of samples in train: {n_train_samples}: {n_control_train} controls, {n_case_train} cases")
    print()

    # Test data
    if test_path is not None:
        # Load test file
        (bim_test, fam_test, bed_test) = read_plink(test_path, verbose=False)
        n_original_test_SNPs = bed.shape[0]
        x_test, y_test = generate_dataset(bed_test, bim_test, fam_test, selected_snp_names)
        x_test, y_test = shuffle(x_test, y_test)
    else:
        # Extract and remove test data from train
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.15, shuffle=True)

    n_test_samples = x_test.shape[0]
    n_test_SNPs = x_test.shape[1]
    print(f"Number of SNPs in test: {n_test_SNPs}")
    n_control_test, n_case_test = count_case_control(y_test)
    print(f"Number of samples in test: {n_test_samples}: {n_control_test} controls, {n_case_test} cases")
    print()

    # Check SNPs
    if n_train_SNPs != n_test_SNPs:
        print("ERROR: Number of train and test SNPs doesn't match")
        exit(1)

    # ----- Deep Neural Network -----
    # Generate and train model
    print("Creating DNN model...")
    DNN_model = create_MLP_model(n_train_SNPs)
    print("Training DNN model...")
    es = EarlyStopping(monitor='val_loss', mode='min', patience=25, restore_best_weights=False, verbose=0)
    history = DNN_model.fit(x_train, y_train, epochs=500, validation_split=0.15, callbacks=[es], verbose=0)
    plot_training_history(history, fold)

    print("Evaluate DNN model...")
    DNN_y_test_prob = DNN_model.predict(x_test)
    DNN_y_test_pred = (DNN_y_test_prob > 0.5).astype("int32")
    DNN_cm = metrics.confusion_matrix(y_test, DNN_y_test_pred)
    DNN_cm_sum += DNN_cm
    DNN_precision = precision_score(y_test, DNN_y_test_pred)
    DNN_precision_list.append(DNN_precision)
    DNN_recall = recall_score(y_test, DNN_y_test_pred)
    DNN_recall_list.append(DNN_recall)
    print(f"DNN \t Precision {DNN_precision:.2f} \t Recall {DNN_recall:.2f} \t for fold {fold}")

    # ----- Support Vector Machine -----
    # Generate and train model
    print("Creating SVM model...")
    SVM_model = SVC(kernel='rbf', probability=True)
    SVM_model.fit(x_train, y_train)

    print("Evaluate SVM model...")
    SVM_y_test_prob = SVM_model.predict_proba(x_test)
    SVM_y_test_prob = SVM_y_test_prob[:, 1]
    SVM_y_test_pred = SVM_model.predict(x_test)
    SVM_cm = metrics.confusion_matrix(y_test, SVM_y_test_pred)
    SVM_cm_sum += SVM_cm
    SVM_precision = precision_score(y_test, SVM_y_test_pred)
    SVM_precision_list.append(SVM_precision)
    SVM_recall = recall_score(y_test, SVM_y_test_pred)
    SVM_recall_list.append(SVM_recall)
    print(f"SVM \t Precision {SVM_precision:.2f} \t Recall {SVM_recall:.2f} \t for fold {fold}")

    # ----- Random Forest -----
    # Generate and train model
    print("Creating RF model...")
    RF_model = RandomForestClassifier()
    RF_model.fit(x_train, y_train)

    print("Evaluate RF model...")
    RF_y_test_prob = RF_model.predict_proba(x_test)
    RF_y_test_prob = RF_y_test_prob[:, 1]
    RF_y_test_pred = RF_model.predict(x_test)
    RF_cm = metrics.confusion_matrix(y_test, RF_y_test_pred)
    RF_cm_sum += RF_cm
    RF_precision = precision_score(y_test, RF_y_test_pred)
    RF_precision_list.append(RF_precision)
    RF_recall = recall_score(y_test, RF_y_test_pred)
    RF_recall_list.append(RF_recall)
    print(f"RF \t Precision {RF_precision:.2f} \t Recall {RF_recall:.2f} \t for fold {fold}")

    # ----- Gradient Boosting -----
    # Generate and train model
    print("Creating GB model...")
    GB_model = GradientBoostingClassifier()
    GB_model.fit(x_train, y_train)

    print("Evaluate GB model...")
    GB_y_test_prob = GB_model.predict_proba(x_test)
    GB_y_test_prob = GB_y_test_prob[:, 1]
    GB_y_test_pred = GB_model.predict(x_test)
    GB_cm = metrics.confusion_matrix(y_test, GB_y_test_pred)
    GB_cm_sum += GB_cm
    GB_precision = precision_score(y_test, GB_y_test_pred)
    GB_precision_list.append(GB_precision)
    GB_recall = recall_score(y_test, GB_y_test_pred)
    GB_recall_list.append(GB_recall)
    print(f"GB \t Precision {GB_precision:.2f} \t Recall {GB_recall:.2f} \t for fold {fold}")

    # ----- Calculate ROC curve ------
    DNN_auc_score = roc_auc_score(y_test, DNN_y_test_prob)
    DNN_fpr, DNN_tpr, _ = roc_curve(y_test, DNN_y_test_prob)
    DNN_tpr = np.interp(base_fpr, DNN_fpr, DNN_tpr)
    DNN_auc_score_list.append(DNN_auc_score)
    DNN_tpr_matrix.append(DNN_tpr)

    SVM_auc_score = roc_auc_score(y_test, SVM_y_test_prob)
    SVM_fpr, SVM_tpr, _ = roc_curve(y_test, SVM_y_test_prob)
    SVM_tpr = np.interp(base_fpr, SVM_fpr, SVM_tpr)
    SVM_auc_score_list.append(SVM_auc_score)
    SVM_tpr_matrix.append(SVM_tpr)

    RF_auc_score = roc_auc_score(y_test, RF_y_test_prob)
    RF_fpr, RF_tpr, _ = roc_curve(y_test, RF_y_test_prob)
    RF_tpr = np.interp(base_fpr, RF_fpr, RF_tpr)
    RF_auc_score_list.append(RF_auc_score)
    RF_tpr_matrix.append(RF_tpr)

    GB_auc_score = roc_auc_score(y_test, GB_y_test_prob)
    GB_fpr, GB_tpr, _ = roc_curve(y_test, GB_y_test_prob)
    GB_tpr = np.interp(base_fpr, GB_fpr, GB_tpr)
    GB_auc_score_list.append(GB_auc_score)
    GB_tpr_matrix.append(GB_tpr)

    # ----- Represent data to 2D -----
    # Reduce to two dimension via Primary Component Analysis
    pca = PCA(n_components=2)
    pca = pca.fit(x_train)
    x_train_2d = pca.transform(x_train)
    x_test_2d = pca.transform(x_test)

    xmax = max(max(x_train_2d[:, 0]), max(x_test_2d[:, 0])) + 0.2
    xmin = min(min(x_train_2d[:, 0]), min(x_test_2d[:, 0])) - 0.2
    ymax = max(max(x_train_2d[:, 1]), max(x_test_2d[:, 1])) + 0.2
    ymin = min(min(x_train_2d[:, 1]), min(x_test_2d[:, 1])) - 0.2

    plot_2d_dataset(x_train_2d, y_train, xmin, xmax, ymin, ymax, "train", fold)
    plot_2d_dataset(x_test_2d, y_test, xmin, xmax, ymin, ymax, "test", fold)

# Plot confusion matrix
plot_confusion_matrix(DNN_cm_sum, 'DNN sum')
plot_confusion_matrix(SVM_cm_sum, 'SVM sum')
plot_confusion_matrix(RF_cm_sum, 'RF sum')
plot_confusion_matrix(GB_cm_sum, 'GB sum')

# Finally plot ROC curve
plot_roc_curve(DNN_auc_score_list, DNN_tpr_matrix, SVM_auc_score_list, SVM_tpr_matrix,
               RF_auc_score_list, RF_tpr_matrix, GB_auc_score_list, GB_tpr_matrix)
