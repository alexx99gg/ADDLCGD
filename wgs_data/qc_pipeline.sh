#!/bin/bash

# -------------------- Apply quality control parameters --------------------
# --chr: chromosome 1 to 22 (discard 23)
# --mind: sample genotyping efficency / call rate
# --geno: marker genotyping efficency / call rate
# --maf: minor allele frecuency
# --hwe: Hardy-Weinberg equilibrium
# --clump-p1: Significance threshold for index SNPs
# --clump-r2: LD threshold for clumping
directory="original/"
file="ADNI1GO2"

# Exclude non-white participants
python3 ../src/exclude_non_white.py "${directory}${file}" ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv ../demographic_data/PTDEMOG.csv "cleaned/${file}"

plink --bfile "cleaned/${file}" --make-bed --missing-genotype N --chr 1-22 --maf 0.01 --geno 0.001 --hwe 0.05 --rel-cutoff 0.25 --out "cleaned/${file}"
# plink --bfile "merged/${file}" --make-bed --missing-genotype N --chr 1-22 --mind 0.1 --maf 0.01 --geno 0.001 --hwe 0.05 --out "cleaned/${file}"

# Generate phenotype data
python3 ../src/generate_pheno.py "cleaned/${file}.fam" ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv

# Split data for k-fold and make association study
#gwas_data_selection_list = ["train_fold", "all", "half_excluded"]
gwas_data_selection="train_fold"

python3 ../src/split_data.py --plink_path "cleaned/${file}" --splits 2 --out_file_folder "subsets${file}_${gwas_data_selection}/" --gwas_data_selection "${gwas_data_selection}"
