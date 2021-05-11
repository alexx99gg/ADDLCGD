import sys

import pandas
from sklearn.utils import shuffle

from read_data import *


def main(argv):
    # read params
    if len(argv) != 4:
        print("Usage: python3 split_data.py fam_file split_ratio out_file1 out_file2")
        exit(2)

    fam_file = argv[0]
    split_ratio = float(argv[1])
    out_file_path1 = argv[2]
    out_file_path2 = argv[3]

    # Read file
    file = pandas.read_csv(fam_file, names=['FID', 'IID', 'father', 'mother', 'sex', 'phenotype'], index_col='IID',
                           delimiter=' ')
    file = shuffle(file)
    headers = file.columns.tolist()

    out_file1 = open(out_file_path1, 'w+')
    out_file2 = open(out_file_path2, 'w+')

    n_samples, _ = file.shape
    # Split data with same label distribution
    cut = n_samples * split_ratio
    i = 0
    for key, data in file.iterrows():
        FID = data[headers.index('FID')]
        IID = key
        if i < cut:
            out_file1.write(f"{FID} {IID}\n")
        else:
            out_file2.write(f"{FID} {IID}\n")

        i += 1


if __name__ == "__main__":
    main(sys.argv[1:])
