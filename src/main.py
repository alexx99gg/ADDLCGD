from pandas_plink import read_plink
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

from plot_utils import *
from read_data import *
from train_data import *

# root_folder = '/content/drive/MyDrive/TFG/'
root_folder = '../'

diagnostic_dict = read_diagnose(file_path=root_folder + 'diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv')

(bim, fam, bed) = read_plink(root_folder + 'wgs_data/cleaned/cleaned_adni_1_2_go_3')

# bed:
#   0 -> First allele
#   1 -> Heterozygous
#   2 -> Second allele
#   math.nan -> missing genotype

n_wgs_samples = bed.shape[1]
n_SNPs = bed.shape[0]

print(f"Number of WGS samples: {n_wgs_samples}")
print(f"Number of variants per WGS sample: {n_SNPs}\n")

IGAP_path = '../wgs_data/cleaned/cleaned_adni_1_2_go_3.assoc'
IGAP_file = pandas.read_csv(IGAP_path, index_col='SNP', delimiter=r"\s+")
IGAP_headers = IGAP_file.columns.tolist()
# Order by p value
IGAP_file = IGAP_file.sort_values(by=["P"], ascending=True)
snp_list = IGAP_file.index
snp_list = snp_list[:700]
print(snp_list)

# Generate dataset from input data
x, y = generate_dataset(diagnostic_dict, bed, bim, fam, snp_list)

n_wgs_samples = x.shape[0]
n_SNPs = x.shape[1]

print(f"Number of WGS selected in dataset: {n_wgs_samples}")
print(f"Number of variants per WGS selected in dataset: {n_SNPs}\n")

alzheimer_cases = np.count_nonzero(y)
no_alzheimer_cases = n_wgs_samples - alzheimer_cases

print(f"Number of Alzheimer's cases in dataset: {alzheimer_cases}")
print(f"Number of NO Alzheimer's cases in dataset: {no_alzheimer_cases}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True)
print(f"Shape of train data: {x_train.shape}")
print(f"Shape of train labels: {y_train.shape}")

print(f"Shape of test data: {x_test.shape}")
print(f"Shape of test labels: {y_test.shape}")

# Create and fit model
print("Creating model...")
model = create_MLP_model(n_SNPs)
print("Creating model... DONE")

print("Training model...")
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)
history = model.fit(x_train, y_train, epochs=100, validation_split=0.111111, callbacks=[es])  # validation_split=0.3
print("Training model... DONE")

plot_training_history(history)

print("Evaluate model...")
model.evaluate(x_test, y_test, verbose=2)

y_test_prob = model.predict(x_test)

plot_confusion_matrix(y_test, y_test_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
plot_roc_curve(fpr, tpr)

auc_score = roc_auc_score(y_test, y_test_prob)
