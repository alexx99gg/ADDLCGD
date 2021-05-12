import sys
from pandas_plink import read_plink
import pandas
from sklearn.utils import shuffle

from read_data import *


def main(argv):
    # read params
    if len(argv) != 5:
        print("Usage: python3 split_data.py plink_file diagnose_file split_ratio out_file1 out_file2")
        exit(2)

    plink_path = argv[0]
    diagnose_path = argv[1]
    split_ratio = float(argv[2])
    out_file_path1 = argv[3]
    out_file_path2 = argv[4]

    # Read file
    fam_file = pandas.read_csv(plink_path + '.fam', names=['FID', 'IID', 'father', 'mother', 'sex', 'phenotype'], index_col='IID',
                           delimiter=' ')
    (bim, fam, bed) = read_plink(plink_path)

    fam_file = shuffle(fam_file)
    headers = fam_file.columns.tolist()

    out_file1 = open(out_file_path1 + '.fam', 'w+')
    out_file2 = open(out_file_path2 + '.fam', 'w+')

    n_samples, _ = fam_file.shape
    # Split data with same label distribution
    diagnostic_dict = read_diagnose(file_path=diagnose_path)
    cut = n_samples * split_ratio
    x, y = generate_dataset(diagnostic_dict, bed, bim, fam, [])
    n_control, n_case = count_case_control(y)

    n_control_cut = n_control * split_ratio
    n_case_cut = n_case * split_ratio
    i_control = 0
    i_case = 0
    for key, data in fam_file.iterrows():
        FID = data[headers.index('FID')]
        IID = key
        diagnose = diagnostic_dict[IID]
        if diagnose == 1:
            # CN
            if i_control < n_control_cut:
                out_file1.write(f"{FID} {IID}\n")
            else:
                out_file2.write(f"{FID} {IID}\n")
            i_control += 1
        elif diagnose == 2:
            # MCI
            out_file1.write(f"{FID} {IID}\n")
        elif diagnose == 3:
            # AD
            if i_case < n_case_cut:
                out_file1.write(f"{FID} {IID}\n")
            else:
                out_file2.write(f"{FID} {IID}\n")
            i_case += 1


if __name__ == "__main__":
    main(sys.argv[1:])
