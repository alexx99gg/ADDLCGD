import os
import subprocess
import sys
from pandas_plink import read_plink1_bin, write_plink1_bin
from sklearn.model_selection import StratifiedKFold

from read_data import *


def main(argv):
    # read params
    if len(argv) != 6:
        print("Usage: python3 split_data.py plink_file diagnose_file splits p1 r2 out_file_root")
        exit(2)

    plink_path = argv[0]
    diagnose_path = argv[1]
    splits = int(argv[2])
    p1 = float(argv[3])
    r2 = float(argv[4])
    out_file_root = argv[5]

    # Read fam file
    fam_file = pandas.read_csv(plink_path + '.fam', names=['FID', 'IID', 'father', 'mother', 'sex', 'phenotype'],
                               index_col='IID', delimiter=' ')
    diagnostic_dict = read_diagnose(file_path=diagnose_path)

    # None
    X = []
    # y label used for the stratification
    y = []

    for key, data in fam_file.iterrows():
        IID = key
        diagnose = diagnostic_dict[IID]
        if diagnose == 1:
            X.append(IID)
            y.append(0)
        elif diagnose == 3:
            X.append(IID)
            y.append(1)
        else:
            continue

    X = np.asarray(X)
    skf = StratifiedKFold(n_splits=splits, shuffle=True)

    G = read_plink1_bin(plink_path + ".bed", plink_path + ".bim", plink_path + ".fam")
    split_i = 0
    for train_index, test_index in skf.split(X, y):
        split_i += 1
        X_train, X_test = X[train_index], X[test_index]
        G_train = G.where(G.iid.isin(X_train), drop=True)
        G_test = G.where(G.iid.isin(X_test), drop=True)

        write_plink1_bin(G_train, f"{out_file_root}_fold_{split_i}_train.bed")
        write_plink1_bin(G_test, f"{out_file_root}_fold_{split_i}_test.bed")

        train_file_path = os.path.abspath(f"{out_file_root}_fold_{split_i}_train")
        test_file_path = os.path.abspath(f"{out_file_root}_fold_{split_i}_test")

        command1 = f"plink --bfile {train_file_path} --assoc fisher-midp --out {train_file_path}"
        command2 = f"plink --bfile {train_file_path} --clump {train_file_path}.assoc.fisher --clump-best --clump-p1 {p1} --clump-r2 {r2} --out {train_file_path}"
        process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process1.communicate()
        print(stdout.decode('utf-8'), stderr.decode('utf-8'))

        process2 = subprocess.Popen(command2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process2.communicate()
        print(stdout.decode('utf-8'), stderr.decode('utf-8'))


if __name__ == "__main__":
    main(sys.argv[1:])
