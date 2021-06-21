import os
import subprocess
import sys
import pandas as pd

from pandas_plink import read_plink1_bin, write_plink1_bin
from sklearn.model_selection import StratifiedKFold

from read_data import *


def main(argv):
    # read params
    if len(argv) != 4:
        print("Usage: python3 split_data.py plink_file splits r2 out_file_root")
        exit(2)

    plink_path = argv[0]
    splits = int(argv[1])
    r2 = float(argv[2])
    out_file_root = argv[3]

    # Read fam file
    fam_file = pd.read_csv(plink_path + '.fam', names=['FID', 'IID', 'father', 'mother', 'sex', 'phenotype'],
                               index_col='IID', delimiter=' ')
    # None
    X = []
    # y label used for the stratification
    y = []

    for key, data in fam_file.iterrows():
        IID = key
        diagnose = data['phenotype']
        if diagnose == 1:
            X.append(IID)
            y.append(0)
        elif diagnose == 2:
            X.append(IID)
            y.append(1)
        else:
            continue

    X = np.asarray(X)
    skf = StratifiedKFold(n_splits=splits, shuffle=True)

    G = read_plink1_bin(plink_path + ".bed", plink_path + ".bim", plink_path + ".fam", verbose=False)
    split_i = 0
    for train_index, test_index in skf.split(X, y):
        split_i += 1
        X_train, X_test = X[train_index], X[test_index]
        G_train = G.where(G.iid.isin(X_train), drop=True)
        G_test = G.where(G.iid.isin(X_test), drop=True)

        write_plink1_bin(G_train, f"{out_file_root}_fold_{split_i}_train.bed", verbose=False)
        write_plink1_bin(G_test, f"{out_file_root}_fold_{split_i}_test.bed", verbose=False)

        train_file_path = os.path.abspath(f"{out_file_root}_fold_{split_i}_train")
        test_file_path = os.path.abspath(f"{out_file_root}_fold_{split_i}_test")

        command1 = f"plink --bfile {train_file_path} --assoc fisher-midp --out {train_file_path}"
        print("Executing command: ", command1)
        process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process1.communicate()
        print(stdout.decode('utf-8'), stderr.decode('utf-8'))

        for p1 in [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]:
            command2 = f"plink --bfile {train_file_path} --clump {train_file_path}.assoc.fisher --clump-best --clump-p1 {p1} --clump-r2 {r2} --allow-no-sex --out {train_file_path}_p1_{p1}"
            print("Executing command: ", command2)
            process2 = subprocess.Popen(command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = process2.communicate()
            print(stdout.decode('utf-8'), stderr.decode('utf-8'))


if __name__ == "__main__":
    main(sys.argv[1:])
