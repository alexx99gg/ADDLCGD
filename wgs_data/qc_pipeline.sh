#!/bin/bash

# -------------------- Combine plink files --------------------
# --chr 1-22: chromosome 1 to 22 (discard 23)
# adni1 + adni2 + adnigo (as in the paper)
plink --merge-list merge_list_adni_1_2_go.txt --make-bed --out merged/adni_1_2_go

plink --merge-list merge_list_adni_2_go.txt --make-bed --out merged/adni_2_go


# adni_1 _ adni2 _ adnigo _ adni3
plink --merge-list merge_list_adni_1_2_go_3.txt --make-bed --out merged/adni_1_2_go_3
# Error! exclude problematic snp
plink --bfile ADNI3_PLINK_Final --flip merged/adni_1_2_go_3-merge.missnp --make-bed --out ADNI3_PLINK_Final_flipped

# -------------------- Apply quality control parameters --------------------
# On the paper:
# --mind 0.1: sample genotyping efficency / call rate
# --geno 0.01: marker genotyping efficency / call rate
# --maf 0.01: minor allele frecuency
# --hwe 0.05: Hardy-Weinberg equilibrium
# --clump-p1 0.001: Significance threshold for index SNPs
# --clump-r2 0.05: LD threshold for clumping

plink --bfile merged/adni_1_2_go_3 --missing-genotype N --make-bed --chr 1-22 --maf 0.01 --geno 0.01 --hwe 0.05 --out cleaned/cleaned_adni_1_2_go_3
plink --bfile merged/adni_1_2_go --missing-genotype N --make-bed --chr 1-22 --maf 0.01 --geno 0.01 --hwe 0.05 --out cleaned/cleaned_adni_1_2_go
plink --bfile merged/adni_2_go --missing-genotype N --make-bed --chr 1-22 --maf 0.01 --geno 0.01 --hwe 0.05 --out cleaned/cleaned_adni_2_go

# Generate phenotype data
python3 ../src/generate_pheno.py merged/adni_1_2_go.fam ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv phenotype_adni_1_2_go.txt
python3 ../src/generate_pheno.py merged/adni_1_2_go_3.fam ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv phenotype_adni_1_2_go_3.txt
# Generate .assoc file
plink --bfile cleaned/cleaned_adni_1_2_go_3 --pheno phenotype_adni_1_2_go_3.txt --assoc --out cleaned/cleaned_adni_1_2_go_3
plink --bfile cleaned/cleaned_adni_1_2_go --pheno phenotype_adni_1_2_go_3.txt --assoc --out cleaned/cleaned_adni_1_2_go
plink --bfile cleaned/cleaned_adni_2_go --pheno phenotype_adni_1_2_go_3.txt --assoc --out cleaned/cleaned_adni_2_go

# Apply LD-clumping
plink --bfile cleaned/cleaned_adni_1_2_go --clump cleaned/cleaned_adni_1_2_go.assoc --clump-best --clump-p1 0.0001 --clump-r2 0.5 --out cleaned/cleaned_adni_1_2_go

plink --bfile cleaned/cleaned_adni_1_2_go --clump ../IGAP_data/IGAP.assoc --clump-best --clump-p1 0.001 --clump-r2 0.05 --out cleaned/cleaned_adni_1_2_go
