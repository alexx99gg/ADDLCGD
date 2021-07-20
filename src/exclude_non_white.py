import sys

import dask
import pandas as pd
from pandas_plink import read_plink1_bin, write_plink1_bin

dask.config.set({"array.slicing.split_large_chunks": False})


def main(argv):
    # read params
    if len(argv) != 4:
        print("Usage: python3 exclude_non_white.py plink_file diagnosis_file demographic_file out_file")
        exit(2)

    plink_path = argv[0]
    diagnosis_path = argv[1]
    demographic_path = argv[2]
    out_path = argv[3]

    # Read files
    fam_file = pd.read_csv(f"{plink_path}.fam", names=['FID', 'IID', 'father', 'mother', 'sex', 'phenotype'],
                           index_col='IID', delimiter=' ')

    diagnosis_file = pd.read_csv(diagnosis_path, index_col='PTID')
    diagnosis_file = diagnosis_file[~diagnosis_file.index.duplicated(keep='last')]

    demographic_file = pd.read_csv(demographic_path, index_col='RID')
    demographic_file = demographic_file[~demographic_file.index.duplicated(keep='first')]

    n_american_native_alaskan_native = 0
    n_asian = 0
    n_hawaiian_native_pacific_islander = 0
    n_black = 0
    n_white = 0
    n_more_than_one = 0
    n_not_known = 0

    IID_to_keep = []

    for IID, data in fam_file.iterrows():
        RID = diagnosis_file.at[IID, 'RID']

        ethnicity = demographic_file.at[RID, 'PTRACCAT']
        if ethnicity == 1:
            n_american_native_alaskan_native += 1
        elif ethnicity == 2:
            n_asian += 1
        elif ethnicity == 3:
            n_hawaiian_native_pacific_islander += 1
        elif ethnicity == 4:
            n_black += 1
        elif ethnicity == 5:
            n_white += 1
            IID_to_keep.append(IID)
        elif ethnicity == 6:
            n_more_than_one += 1
        else:
            n_not_known += 1

    print(f"Number of American Indian or Alaskan Native subjects: {n_american_native_alaskan_native}")
    print(f"Number of Asian subjects: {n_asian}")
    print(f"Number of Native Hawaiian or Other Pacific Islander subjects: {n_hawaiian_native_pacific_islander}")
    print(f"Number of Black subjects: {n_black}")
    print(f"Number of White subjects: {n_white}")
    print(f"Number of More than one race subjects: {n_more_than_one}")
    print(f"Number of Not known race subjects: {n_not_known}")

    G = read_plink1_bin(f"{plink_path}.bed", plink_path + ".bim", plink_path + ".fam", verbose=False)

    G = G.where(G.iid.isin(IID_to_keep), drop=True)
    print(f"Writting new plink file with only White subjects to {out_path}")
    write_plink1_bin(G, f"{out_path}.bed", verbose=False)


if __name__ == "__main__":
    main(sys.argv[1:])
