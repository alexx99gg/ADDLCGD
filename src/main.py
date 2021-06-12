import numpy as np
from pandas_plink import read_plink
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping

from plot_utils import *
from read_data import *
from train_data import *

plt.rcParams['font.size'] = '16'

dataset = "ADNI1GO2"

diagnose_path = "../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv"
clumpd_1_path = f"../wgs_data/subsets/{dataset}_1.clumped"
subset_1_path = f"../wgs_data/subsets/{dataset}_1"
subset_2_path = f"../wgs_data/subsets/{dataset}_2"

print(f"Reading dataset {dataset}")

# Read diagnoses
diagnostic_dict = read_diagnose(file_path=diagnose_path)

# Get SNPs to keep
clump_file = pandas.read_csv(clumpd_1_path, index_col='SNP', delimiter=r" +")
clump_headers = clump_file.columns.tolist()
# Order by P value
clump_file = clump_file.sort_values(by=["P"], ascending=True)
snp_list = clump_file.index.tolist()
# p_snp = clump_file['PSNP'].tolist()
# p_snp = [p for p in p_snp if str(p) != 'nan']

# snp_list = snp_list + p_snp
# Get the first ones
# snp_list = snp_list[:5]
print(f"{len(snp_list)} SNPs selected (in relevance order):")
print(snp_list)

# Load train data
(bim, fam, bed) = read_plink(subset_1_path)
n_original_train_SNPs = bed.shape[0]
print(f"Number of original SNPs in train: {n_original_train_SNPs}\n")

x_train, y_train = generate_dataset(diagnostic_dict, bed, bim, fam, snp_list)

x_train, y_train = shuffle(x_train, y_train)

n_train_samples = x_train.shape[0]
n_train_SNPs = x_train.shape[1]

print(f"Number of SNPs in train: {n_train_SNPs}")
print(f"Number of samples in train: {n_train_samples}")
n_control_train, n_case_train = count_case_control(y_train)
print(f"Number of control in train: {n_control_train}, number of cases in train: {n_case_train}\n")
print(f"Percentage of controls in train: {n_control_train / n_train_samples:.2f}")
print(f"Percentage of cases in train: {n_case_train / n_train_samples:.2f}")
print()

# Test data
if subset_2_path is not None:
    (bim_test, fam_test, bed_test) = read_plink(subset_2_path)
    n_original_test_SNPs = bed.shape[0]
    print(f"Number of original SNPs in test: {n_original_test_SNPs}\n")

    x_test, y_test = generate_dataset(diagnostic_dict, bed_test, bim_test, fam_test, snp_list)
    x_test, y_test = shuffle(x_test, y_test)
else:
    # Extract and remove test data from train
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, shuffle=True)

n_test_samples = x_test.shape[0]
n_test_SNPs = x_test.shape[1]
print(f"Number of SNPs in test: {n_test_SNPs}")
print(f"Number of samples in test: {n_test_samples}")
n_control_test, n_case_test = count_case_control(y_test)
print(f"Number of control in test: {n_control_test}, number of cases in test: {n_case_test}\n")
print(f"Percentage of controls in test: {n_control_test / n_test_samples:.2f}")
print(f"Percentage of cases in test: {n_case_test / n_test_samples:.2f}")
print()

if n_train_SNPs != n_test_SNPs:
    print("ERROR: Number of train and test SNPs doesn't match")
    exit(1)

# ----- Deep Neural Network -----
# Generate and train model
print("Creating DNN model...")
DNN_model = create_MLP_model(n_train_SNPs)
print("Creating model... DONE")
print("Training DNN model...")
es = EarlyStopping(monitor='val_auc', mode='max', patience=25, restore_best_weights=True, verbose=1)
history = DNN_model.fit(x_train, y_train, epochs=500, validation_split=0.15, callbacks=[es])
print("Training DNN model... DONE")
plot_training_history(history)

print("Evaluate DNN model...")
DNN_y_test_prob = DNN_model.predict(x_test)
plot_confusion_matrix(y_test, DNN_y_test_prob, "DNN")

# ----- Support Vector Machine -----
# Generate and train model
print("Creating SVC model...")
SVC_model = SVC(kernel='rbf')
SVC_model.fit(x_train, y_train)

print("Evaluate SVC model...")
SVC_y_test_prob = SVC_model.predict(x_test)
plot_confusion_matrix(y_test, SVC_y_test_prob, "SVC")

# ----- Random Forest -----
# Generate and train model
print("Creating RF model...")
RF_model = RandomForestClassifier()
RF_model.fit(x_train, y_train)

print("Evaluate RF model...")
RF_y_test_prob = RF_model.predict(x_test)
plot_confusion_matrix(y_test, RF_y_test_prob, 'RF')

# ----- Gradient Boosting -----
# Generate and train model
print("Creating GBC model...")
GBC_model = GradientBoostingClassifier()
GBC_model.fit(x_train, y_train)

print("Evaluate GBC model...")
GBC_y_test_prob = GBC_model.predict(x_test)
plot_confusion_matrix(y_test, GBC_y_test_prob, 'GBC')


# ----- Plot ROC curve ------
plot_roc_curve(y_test, DNN_y_test_prob, SVC_y_test_prob, RF_y_test_prob, GBC_y_test_prob)

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

plot_2d_dataset(x_train_2d, y_train, xmin, xmax, ymin, ymax)
plot_2d_dataset(x_test_2d, y_test, xmin, xmax, ymin, ymax)
