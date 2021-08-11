import argparse
import os
import subprocess

from pandas_plink import read_plink1_bin, write_plink1_bin
from sklearn.model_selection import StratifiedKFold

from read_data import *

r2 = 0.25


def execute_command(command):
    print("Executing command: ", command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    print(stdout.decode('utf-8'), stderr.decode('utf-8'))


def k_fold(G, X, y, folds, out_file_folder, do_assoc_with_train=False):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=23)

    fold_i = 0
    for train_index, test_index in skf.split(X, y):
        fold_i += 1
        X_train, X_test = X[train_index], X[test_index]
        G_train = G.where(G.iid.isin(X_train), drop=True)
        G_test = G.where(G.iid.isin(X_test), drop=True)

        # Write train and test to plink
        write_plink1_bin(G_train, f"{out_file_folder}fold_{fold_i}_train.bed", verbose=False)
        write_plink1_bin(G_test, f"{out_file_folder}fold_{fold_i}_test.bed", verbose=False)

        if do_assoc_with_train:
            # Association study with train data for each fold
            train_file_path = os.path.abspath(f"{out_file_folder}fold_{fold_i}_train")
            assoc_command = f"plink --bfile {train_file_path} --assoc fisher-midp --out {train_file_path}"
            execute_command(assoc_command)
            # Clumping
            for p1 in [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]:
                clump_command = f"plink --bfile {train_file_path} --clump {train_file_path}.assoc.fisher --clump-best --clump-p1 {p1:.0e} --clump-r2 {r2} --allow-no-sex --out {train_file_path}_p1_{p1:.0e}"
                execute_command(clump_command)


def main(args):
    # read params
    plink_path = args.plink_path
    folds = args.folds
    out_file_folder = args.out_file_folder
    selection_method = args.selection_method

    if not os.path.exists(out_file_folder):
        os.makedirs(out_file_folder)

    # Read each patient IID and diagnose from .fam file
    fam_file = pd.read_csv(plink_path + '.fam', names=['FID', 'IID', 'father', 'mother', 'sex', 'phenotype'],
                           index_col='IID', delimiter=' ')
    # X data with IID
    X = []
    # y label used with diagnose used for the stratification
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

    # Read dataframe from PLINK file
    G = read_plink1_bin(plink_path + ".bed", plink_path + ".bim", plink_path + ".fam", verbose=False)

    if selection_method == "leakage":
        # Do association study for whole dataset
        # WARNING: no separation between train and test
        assoc_command = f"plink --bfile {plink_path} --assoc fisher-midp --out {out_file_folder}all"
        execute_command(assoc_command)
        # Clumping
        for p1 in [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]:
            clump_command = f"plink --bfile {plink_path} --clump {out_file_folder}all.assoc.fisher --clump-best --clump-p1 {p1:.0e} --clump-r2 {r2} --allow-no-sex --out {out_file_folder}all_p1_{p1:.0e}"
            execute_command(clump_command)
        k_fold(G, X, y, folds, out_file_folder, do_assoc_with_train=False)

    elif selection_method == "split":
        # Exclude part of the data to perform GWAS and use the rest is used for train/test

        gwas_ratio = float(args.gwas_ratio)
        if gwas_ratio == 0.5:
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=23)
        elif gwas_ratio == 0.75:
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=23)
        elif gwas_ratio == 0.8:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
        elif gwas_ratio == 0.9:
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)
        else:
            print(f"ERROR: gwas_ratio value must be 0.5 0.75 0.8 or 0.9")

        gwas_index, train_test_index = next(skf.split(X, y))

        X_gwas, X_train_test = X[gwas_index], X[train_test_index]
        y_gwas, y_train_test = y[gwas_index], y[train_test_index]
        G_gwas = G.where(G.iid.isin(X_gwas), drop=True)
        G_train_test = G.where(G.iid.isin(X_train_test), drop=True)

        write_plink1_bin(G_gwas, f"{out_file_folder}gwas.bed", verbose=False)
        command1 = f"plink --bfile {out_file_folder}gwas --assoc fisher-midp --out {out_file_folder}gwas"
        execute_command(command1)
        # Clumping
        for p1 in [5e-8, 1e-5, 1e-4, 1e-3, 1e-2]:
            command2 = f"plink --bfile {out_file_folder}gwas --clump {out_file_folder}gwas.assoc.fisher --clump-best --clump-p1 {p1:.0e} --clump-r2 {r2} --allow-no-sex --out {out_file_folder}gwas_p1_{p1:.0e}"
            execute_command(command2)

        k_fold(G_train_test, X_train_test, y_train_test, folds, out_file_folder, do_assoc_with_train=False)

    elif selection_method == "train_fold":
        k_fold(G, X, y, folds, out_file_folder, do_assoc_with_train=True)
    # Divide the rest in k-folds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plink_path", help="path to find plink file")

    parser.add_argument("--selection_method",
                        help="Selection method for GWAS/train data: \"leakage\", \"split\", \"train_fold\"")

    parser.add_argument("--folds",
                        help="number of folds to split the data in train/test (necessary for \"train_fold\ mode)",
                        type=int, default=5)
    parser.add_argument("--gwas_ratio", help="Part excluded to perform GWAS study (necessary for \"split\" mode)")

    parser.add_argument("--out_file_folder", help="path to output file")
    args = parser.parse_args()
    main(args)
