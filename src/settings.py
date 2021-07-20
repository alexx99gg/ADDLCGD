import os

dataset = "ADNI1GO2"

p1_list = [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]
p1 = 1e-4

gwas_data_selection_list = ["train_fold", "all", "part_excluded"]
gwas_data_selection = "train_fold"

dataset_folder = f"../wgs_data/subsets{dataset}_{gwas_data_selection}/"

save_dir = f"../results/dataset_{dataset}_p1_{p1}_selection_{gwas_data_selection}/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
