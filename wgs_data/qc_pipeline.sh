#!/bin/bash

# -------------------- Combine plink files --------------------
# --chr 1-22: chromosome 1 to 22 (discard 23)
# adni1 + adnigo + adni2 (as in the paper)
plink --merge-list merge_list_ADNI1GO2.txt --make-bed --out merged/ADNI1GO2

plink --merge-list merge_list_ADNIGO2.txt --make-bed --out merged/ADNIGO2


# adni_1 _ adni2 _ adnigo _ adni3
plink --merge-list merge_list_ADNI1GO23.txt --make-bed --out merged/ADNI1GO23
# Error! exclude problematic snp
plink --bfile original/ADNI3 --exclude merged/ADNI12GO3-merge.missnp --make-bed --out original/ADNI3_fixSNP

# -------------------- Apply quality control parameters --------------------
# On the paper:
# --chr 1-22: chromosome 1 to 22 (discard 23)
# --mind 0.1: sample genotyping efficency / call rate
# --geno 0.01: marker genotyping efficency / call rate
# --maf 0.01: minor allele frecuency
# --hwe 0.05: Hardy-Weinberg equilibrium
# --clump-p1 0.001: Significance threshold for index SNPs
# --clump-r2 0.05: LD threshold for clumping
file="ADNI1GO2"

plink --bfile "original/${file}" --make-bed --missing-genotype N --chr 1-22 --rel-cutoff 0.05 --out "cleaned/${file}"
plink --bfile "cleaned/${file}" --make-bed --missing-genotype N --maf 0.05 --geno 0.001 --hwe 0.05 --out "cleaned/${file}"
# plink --bfile "merged/${file}" --make-bed --missing-genotype N --chr 1-22 --mind 0.1 --maf 0.01 --geno 0.001 --hwe 0.05 --out "cleaned/${file}"


# Split data
python3 ../src/split_data.py "cleaned/${file}" ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv 0.9 "subsets/${file}_1" "subsets/${file}_2"
plink --bfile "cleaned/${file}" --keep "subsets/${file}_1.fam" --make-bed --out "subsets/${file}_1"
plink --bfile "cleaned/${file}" --keep "subsets/${file}_2.fam" --make-bed --out "subsets/${file}_2"

# Generate phenotype data
python3 ../src/generate_pheno.py "subsets/${file}_1.fam" ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv "../phenotype_data/${file}_1.txt"

# Generate .assoc file of subset 1
plink --bfile "subsets/${file}_1" --pheno "../phenotype_data/${file}_1.txt" --assoc fisher-midp perm --out "subsets/${file}_1"

# Apply LD-clumping
plink --bfile "subsets/${file}_1" --clump "subsets/${file}_1.assoc.fisher.perm" --clump-field EMP1 --clump-best --clump-p1 0.0000075 --clump-r2 0.5 --out "subsets/${file}_1"
