from pandas_plink import read_plink

from read_data import *
from train_data import *

diagnostic_dict = read_diagnose(file_path='../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv')

(bim, fam, bed) = read_plink('../wgs_data/clean')

# print(".fam file:")
# print(fam)
# print("")
# print(".bed file:")
# print(bed.compute())
# print("")

n_wgs_samples = bed.shape[1]
n_snps = bed.shape[0]

print(f"Current number of WGS samples: {n_wgs_samples}\n")
print(f"Current number of variants per WGS sample: {n_snps}\n")


x, y = generate_dataset(diagnostic_dict, fam, bed)

model = create_model(n_snps)

model.fit(x, y, epochs=5)

model.evaluate(x,  y, verbose=2)
