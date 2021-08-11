import os

dataset = "ADNI1GO2"

p1_list = [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]
p1 = 5e-8

selection_method_list = ["train_fold", "leakage", "split"]
selection_method = "leakage"

dataset_folder = f"../wgs_data/processed/{dataset}-selection_method_{selection_method}/"
save_dir = f"../results/dataset_{dataset}-p1_{p1:.0e}-selection_method_{selection_method}/"

if selection_method == "split":
    gwas_ratio = 0.8
    dataset_folder = f"../wgs_data/processed/{dataset}-selection_method_{selection_method}-gwas_ratio_{gwas_ratio}/"
    save_dir = f"../results/dataset_{dataset}-p1_{p1:.0e}-selection_method_{selection_method}-gwas_ratio_{gwas_ratio}/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
