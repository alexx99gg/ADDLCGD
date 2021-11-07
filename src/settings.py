import os

from snps_lists.snps_lists import snpedia_snp_list
from snps_lists.snps_lists import PMC4876682_snp_list

dataset = "ADNI1GO2"


p1_list = [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]
p1 = 1e-4

selection_method_list = ["train_fold", "leakage", "split", "external_study"]
selection_method = "external_study"

dataset_folder = f"../wgs_data/processed/{dataset}-selection_method_{selection_method}/"
save_dir = f"../results/dataset_{dataset}-p1_{p1:.0e}-selection_method_{selection_method}/"

if selection_method == "split":
    gwas_ratio = 0.8
    dataset_folder = f"../wgs_data/processed/{dataset}-selection_method_{selection_method}-gwas_ratio_{gwas_ratio}/"
    save_dir = f"../results/dataset_{dataset}-p1_{p1:.0e}-selection_method_{selection_method}-gwas_ratio_{gwas_ratio}/"
elif selection_method == "external_study":
    snp_source_list = ["PMC4876682_snp_list", "snpedia_snp_list", "tfg_eduardo_snp_list"]
    snp_source = "tfg_eduardo_snp_list"
    dataset_folder = f"../wgs_data/processed/{dataset}-selection_method_leakage/"
    save_dir = f"../results/dataset_{dataset}-selection_method_{selection_method}-snp_source_{snp_source}/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
