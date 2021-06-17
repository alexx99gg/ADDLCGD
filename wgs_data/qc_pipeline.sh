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

plink --bfile "${directory}${file}" --make-bed --missing-genotype N --chr 1-22  --out "cleaned/${file}"
plink --bfile "cleaned/${file}" --make-bed --missing-genotype N --rel-cutoff 0.05 --out "cleaned/${file}"
plink --bfile "cleaned/${file}" --make-bed --missing-genotype N --maf 0.01 --geno 0.01 --hwe 0.05 --out "cleaned/${file}"
# plink --bfile "merged/${file}" --make-bed --missing-genotype N --chr 1-22 --mind 0.1 --maf 0.01 --geno 0.001 --hwe 0.05 --out "cleaned/${file}"

# Generate phenotype data
python3 ../src/generate_pheno.py "cleaned/${file}.fam" ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv

# Split data
python3 ../src/split_data.py "cleaned/${file}" 5 1e-5 0.5 "subsets/${file}"
