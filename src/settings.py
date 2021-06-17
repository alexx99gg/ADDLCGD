import os

dataset_path = "../wgs_data/subsets/"
dataset = "ADNI1GO2"

p1_list = [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]
p1 = 1e-4

save_dir = f"../results/dataset_{dataset}_p1_{p1}/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
