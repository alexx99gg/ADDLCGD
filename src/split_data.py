import os
import subprocess
import sys

import argparse

import numpy as np
from pandas_plink import read_plink1_bin, write_plink1_bin
from sklearn.model_selection import StratifiedKFold

from read_data import *


def execute_command(command):
    print("Executing command: ", command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    print(stdout.decode('utf-8'), stderr.decode('utf-8'))


def main(args):
    # read params
    plink_path = args.plink_path
    splits = args.splits
    r2 = 0.25
    out_file_folder = args.out_file_folder
    gwas_data_selection = args.gwas_data_selection

    if not os.path.exists(out_file_folder):
        os.makedirs(out_file_folder)

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
    y = np.asarray(y)
    G = read_plink1_bin(plink_path + ".bed", plink_path + ".bim", plink_path + ".fam", verbose=False)


    if gwas_data_selection == "all":
        # Do association study for whole dataset
        # WARNING: no separation between train and test
        command1 = f"plink --bfile {plink_path} --assoc fisher-midp --out {plink_path}"
        execute_command(command1)
        # Clumping
        for p1 in [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]:
            command2 = f"plink --bfile {plink_path} --clump {plink_path}.assoc.fisher --clump-best --clump-p1 {p1} --clump-r2 {r2} --allow-no-sex --out {plink_path}_p1_{p1}"
            execute_command(command2)
    
    elif gwas_data_selection == "half_excluded":
        # Exlude 50% of data to perform GWAS
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=23)

        gwas_index, train_test_index = next(skf.split(X, y))

        X_gwas, X_train_test = X[gwas_index], X[train_test_index]
        y_gwas, y_train_test = y[gwas_index], y[train_test_index]
        G_gwas = G.where(G.iid.isin(X_gwas), drop=True)

        write_plink1_bin(G_gwas, f"{out_file_folder}half_excluded.bed", verbose=False)
        command1 = f"plink --bfile {out_file_folder}half_excluded --assoc fisher-midp --out {out_file_folder}half_excluded"
        execute_command(command1)
        # Clumping
        for p1 in [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]:
            command2 = f"plink --bfile {out_file_folder}half_excluded --clump {out_file_folder}half_excluded.assoc.fisher --clump-best --clump-p1 {p1} --clump-r2 {r2} --allow-no-sex --out {out_file_folder}half_excluded_p1_{p1}"
            execute_command(command2)

        X = X_train_test
        y = y_train_test

    # Create n_splits
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=23)
    split_i = 0

    for train_index, test_index in skf.split(X, y):
        split_i += 1
        X_train, X_test = X[train_index], X[test_index]
        G_train = G.where(G.iid.isin(X_train), drop=True)
        G_test = G.where(G.iid.isin(X_test), drop=True)

        write_plink1_bin(G_train, f"{out_file_folder}fold_{split_i}_train.bed", verbose=False)
        write_plink1_bin(G_test, f"{out_file_folder}fold_{split_i}_test.bed", verbose=False)

        if gwas_data_selection == "train_fold":
            # Association study with train data for each fold

            train_file_path = os.path.abspath(f"{out_file_folder}fold_{split_i}_train")
            command1 = f"plink --bfile {train_file_path} --assoc fisher-midp --out {train_file_path}"
            execute_command(command1)
            # Clumping
            for p1 in [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]:
                command2 = f"plink --bfile {train_file_path} --clump {train_file_path}.assoc.fisher --clump-best --clump-p1 {p1} --clump-r2 {r2} --allow-no-sex --out {train_file_path}_p1_{p1}"
                execute_command(command2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plink_path", help="path to find plink file")
    parser.add_argument("-s", "--splits", help="number of splits to split train/test the data", type=int)
    parser.add_argument("-m", "--gwas_data_selection", help="GWAS data: \"all\", \"half_excluded\", \"train_fold\"")
    parser.add_argument("-o", "--out_file_folder", help="path to output file")
    args = parser.parse_args()
    main(args)
