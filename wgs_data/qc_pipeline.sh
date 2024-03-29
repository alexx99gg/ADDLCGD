#!/bin/bash

# -------------------- Apply quality control parameters --------------------
# --chr: chromosome 1 to 22 (discard 23)
# --mind: sample genotyping efficency / call rate
# --geno: marker genotyping efficency / call rate
# --maf: minor allele frecuency
# --hwe: Hardy-Weinberg equilibrium
# --clump-p1: Significance threshold for index SNPs
# --clump-r2: LD threshold for clumping
directory="merged/"
file="ADNI1GO23"

# Exclude non-white participants
python3 ../src/exclude_non_white.py "${directory}${file}" ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv ../demographic_data/PTDEMOG.csv "cleaned/${file}"

plink --bfile "cleaned/${file}" --make-bed --missing-genotype N --chr 1-22 --maf 0.01 --geno 0.001 --hwe 0.05 --rel-cutoff 0.25 --out "cleaned/${file}"
# plink --bfile "merged/${file}" --make-bed --missing-genotype N --chr 1-22 --mind 0.1 --maf 0.01 --geno 0.001 --hwe 0.05 --out "cleaned/${file}"

# Generate phenotype data
python3 ../src/generate_pheno.py "cleaned/${file}.fam" ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv

# Split data for k-fold and make association study
#selection_method_list = ["train_fold", "leakage", "split"]

python3 ../src/split_data.py --plink_path "cleaned/${file}" --selection_method train_fold --folds 5 --out_file_folder "processed/${file}-selection_method_train_fold/"
python3 ../src/split_data.py --plink_path "cleaned/${file}" --selection_method split --gwas_ratio 0.8 --folds 5 --out_file_folder "processed/${file}-selection_method_split-gwas_ratio_0.8/"
python3 ../src/split_data.py --plink_path "cleaned/${file}" --selection_method split --gwas_ratio 0.5 --folds 5 --out_file_folder "processed/${file}-selection_method_split-gwas_ratio_0.5/"
python3 ../src/split_data.py --plink_path "cleaned/${file}" --selection_method leakage --folds 5 --out_file_folder "processed/${file}-selection_method_leakage/"
