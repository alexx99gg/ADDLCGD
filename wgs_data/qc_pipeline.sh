#!/bin/bash

# -------------------- Combine plink files --------------------
# --chr 1-22: chromosome 1 to 22 (discard 23)
# adni1 + adni2 + adnigo (as in the paper)
plink --merge-list merge-list-adni1+2+go.txt --chr 1-22 --make-bed --out adni1+2+go

# adni1 + adni2 + adnigo + adni3
plink --merge-list merge-list-adni1+2+go+3.txt --geno 0.001 --chr 1-22 --make-bed --out adni1+2+go+3
# Error! exclude problematic snp
plink --bfile ADNI3_PLINK_Final --exclude adni1+2+go+3-merge.missnp --make-bed --out ADNI3_PLINK_Final_excluded

# -------------------- Apply quality control parameters --------------------
# On the paper:
# --mind 0.1: sample genotyping efficency / call rate
# --geno 0.01: marker genotyping efficency / call rate
# --maf 0.01: minor allele frecuency
# --hwe 0.05: Hardy-Weinberg equilibrium
# --clump-p1 0.001: Significance threshold for index SNPs
# --clump-r2 0.05: LD threshold for clumping

plink --bfile adni1+2+go --missing-genotype N --make-bed --mind 0.3 --maf 0.00001 --geno 0.001 --hwe 0.05 --out clean_adni1+2+go

# Generate phenotype data
python3 ../src/generate_pheno.py ../wgs_data/adni1+2+go.fam ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv ../wgs_data/adni1+2+go_pheno.txt
# Generate .assoc file
plink --bfile clean_adni1+2+go --pheno adni1+2+go_pheno.txt --assoc --out clean_adni1+2+go
# Apply LD-clumping
plink --bfile clean_adni1+2+go --clump clean_adni1+2+go.assoc --clump-best --clump-p1 0.0001 --clump-r2 0.5 --out clean_adni1+2+go