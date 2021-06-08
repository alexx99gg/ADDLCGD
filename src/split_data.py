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
    fam_file = pandas.read_csv(plink_path + '.fam', names=['FID', 'IID', 'father', 'mother', 'sex', 'phenotype'],
                               index_col='IID',
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
    x, y = generate_dataset(diagnostic_dict, bed, bim, fam, ['a'])
    n_control, n_case = count_case_control(y)

    n_control_1 = int(n_control * split_ratio)
    n_case_1 = int(n_case * split_ratio)

    n_control_2 = n_control - n_control_1
    n_case_2 = n_case - n_case_1

    if n_control > n_case:
        n_control_1 = n_case_1
        n_control_2 = n_case_2
    elif n_case > n_control:
        n_case_1 = n_control_1
        n_case_2 = n_control_2

    counter_control_1, counter_control_2 = 0, 0
    counter_case_1, counter_case_2 = 0, 0

    for key, data in fam_file.iterrows():
        FID = data[headers.index('FID')]
        IID = key
        diagnose = diagnostic_dict[IID]
        if diagnose == 1:
            # CN
            if counter_control_1 < n_control_1:
                out_file1.write(f"{FID} {IID}\n")
                counter_control_1 += 1
            elif counter_control_2 < n_control_2:
                out_file2.write(f"{FID} {IID}\n")
                counter_control_2 += 1
        elif diagnose == 2:
            # MCI discard
            pass
        elif diagnose == 3:
            # AD
            if counter_case_1 < n_case_1:
                out_file1.write(f"{FID} {IID}\n")
                counter_case_1 += 1
            elif counter_case_2 < n_case_2:
                out_file2.write(f"{FID} {IID}\n")
                counter_case_2 += 1
    print(f"Number of control in dataset 1: {n_control_1}, in dataset 2: {n_control_2}")
    print(f"Number of cases in dataset 1: {n_case_1}, in dataset 2: {n_case_2}")

    out_file1.close()
    out_file2.close()


if __name__ == "__main__":
    main(sys.argv[1:])
