#!/bin/bash

# -------------------- Combine plink files --------------------
# --chr 1-22: chromosome 1 to 22 (discard 23)
# adni1 + adni2 + adnigo (as in the paper)
plink --merge-list merged/merge_list_ADNI12GO.txt --make-bed --out merged/ADNI12GO


# adni_1 _ adni2 _ adnigo _ adni3
plink --merge-list merged/merge_list_ADNI12GO3.txt --make-bed --out merged/ADNI12GO3
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
file="ADNI12GO"

plink --bfile "merged/${file}" --missing-genotype N --make-bed --chr 1-22 --maf 0.01 --geno 0.01 --hwe 0.05 --out "cleaned/${file}"

# Generate phenotype data
python3 ../src/generate_pheno.py "cleaned/${file}.fam" ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv "../phenotype_data/${file}.txt"

# Split data
python3 ../src/split_data.py "cleaned/${file}" ../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv 0.8 "subsets/${file}_1" "subsets/${file}_2"
plink --bfile "cleaned/${file}" --keep "subsets/${file}_1.fam" --make-bed --out "subsets/${file}_1"
plink --bfile "cleaned/${file}" --keep "subsets/${file}_2.fam" --make-bed --out "subsets/${file}_2"

# Generate .assoc file of subset 1
plink --bfile "subsets/${file}_1" --pheno "../phenotype_data/${file}.txt" --assoc --out "subsets/${file}_1"

# Apply LD-clumping
plink --bfile "subsets/${file}_1" --clump "subsets/${file}_1.assoc" --clump-best --clump-p1 0.001 --clump-r2 0.05 --out "subsets/${file}_1"

