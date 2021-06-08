from pandas_plink import read_plink
from sklearn.metrics import roc_curve, roc_auc_score, auc
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
# snp_list = snp_list[:15]
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
print("Creating DNNL model...")
DNNL_model = create_MLP_model(n_train_SNPs)
print("Creating model... DONE")
print("Training DNNL model...")
es = EarlyStopping(monitor='val_auc', mode='max', patience=50, restore_best_weights=True, verbose=1)
history = DNNL_model.fit(x_train, y_train, epochs=500, validation_split=0.15, callbacks=[es])
print("Training DNNL model... DONE")
plot_training_history(history)

print("Evaluate DNNL model...")
DNNL_model.evaluate(x_test, y_test, verbose=2)

DNNL_y_test_prob = DNNL_model.predict(x_test)
DNNL_fpr, DNNL_tpr, DNNL_thresholds = roc_curve(y_test, DNNL_y_test_prob)
auc_score = roc_auc_score(y_test, DNNL_y_test_prob)

print(f"DNNL model AUC in test data: {auc_score}")

plot_confusion_matrix(y_test, DNNL_y_test_prob)

plot_roc_curve(DNNL_fpr, DNNL_tpr)

# ----- Support Vector Machine -----
# Generate and train model
print("Creating SVC model...")
SVC_model = SVC(kernel='rbf')
SVC_model.fit(x_train, y_train)

score_train = SVC_model.score(x_train, y_train)
print(f"SVC model mean accuracy in train data: {score_train}")

score_test = SVC_model.score(x_test, y_test)

SVC_y_test_prob = SVC_model.predict(x_test)
SVC_fpr, SVC_tpr, SVC_thresholds = roc_curve(y_test, SVC_y_test_prob)
auc_score = roc_auc_score(y_test, SVC_y_test_prob)

print(f"SVC model mean accuracy in test data: {score_test}")
print(f"SVC model AUC in test data: {auc_score}")
plot_confusion_matrix(y_test, SVC_y_test_prob)
plot_roc_curve(SVC_fpr, SVC_tpr)

# ----- Represent data to 2D -----
# Reduce to two dimension via Primary Component Analysis
pca = PCA(n_components=2)
pca = pca.fit(x_train)
x_train_2d = pca.transform(x_train)
x_test_2d = pca.transform(x_test)

plot_2d_dataset(x_train_2d, y_train)
plot_2d_dataset(x_test_2d, y_test)
