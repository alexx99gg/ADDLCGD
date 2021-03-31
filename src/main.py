from pandas_plink import read_plink
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from read_data import *
from train_data import *
from utils import *

diagnostic_dict = read_diagnose(file_path='../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv')

(bim, fam, bed) = read_plink('../wgs_data/clean')

n_wgs_samples = bed.shape[1]
n_SNPs = bed.shape[0]

print(f"Current number of WGS samples: {n_wgs_samples}\n")
print(f"Current number of variants per WGS sample: {n_SNPs}\n")

# Generate dataset from input data
x, y = generate_dataset(diagnostic_dict, fam, bed)
# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True)
print(f"Shape of train data: {x_train.shape}")
print(f"Shape of train labels: {y_train.shape}")

print(f"Shape of test data: {x_test.shape}")
print(f"Shape of test labels: {y_test.shape}")
# print(y_test)

# Create and fit model
model = create_model(n_SNPs)
model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test, y_test, verbose=2)

y_test_prob = model.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
plot_roc_curve(fpr, tpr)

auc_score = roc_auc_score(y_test, y_test_prob)
