#!/bin/bash

# -------------------- Combine plink files --------------------
plink --merge-list merge_list_ADNI1GO23.txt --make-bed --out merged/ADNI1GO23

plink --merge-list merge_list_ADNIGO2.txt --make-bed --out merged/ADNIGO2

plink --merge-list merge_list_ADNI1GO2.txt --make-bed --out merged/ADNI1GO2

plink --merge-list merge_list_ADNIALL.txt --chr 1-22 --make-bed --out merged/ADNIALL


# adni1 + adni2 + adnigo + adni3
plink --merge-list merge_list_ADNI1GO23.txt --make-bed --out merged/ADNI1GO23
# Error! exclude problematic snp
plink --bfile original/ADNI3 --exclude merged/ADNI12GO3-merge.missnp --make-bed --out original/ADNI3_fixSNP