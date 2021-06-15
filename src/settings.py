import os

dataset_path = "../wgs_data/subsets/"
dataset = "ADNI1GO23"

n_SNPs = 15

save_dir = f"../results/dataset_{dataset}_snps_{n_SNPs}/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
