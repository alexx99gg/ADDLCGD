#!/bin/bash

# -------------------- Combine plink files --------------------
# adni1 + adnigo + adni2
plink --merge-list merge_list_ADNI1GO2.txt --make-bed --out merged/ADNI1GO2

plink --merge-list merge_list_ADNIGO2.txt --make-bed --out merged/ADNIGO2


# adni1 + adni2 + adnigo + adni3
plink --merge-list merge_list_ADNI1GO23.txt --make-bed --out merged/ADNI1GO23
# Error! exclude problematic snp
plink --bfile original/ADNI3 --exclude merged/ADNI12GO3-merge.missnp --make-bed --out original/ADNI3_fixSNP