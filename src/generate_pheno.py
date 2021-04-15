import sys

import pandas

from read_data import *

def main(argv):
    # read params
    if len(argv) != 3:
        print("Usage: python3 generate_pheno.py fam_file diagnosis_file pheno_file")
        exit(2)

    fam_file = argv[0]
    diagnosis_file = argv[1]
    pheno_file = argv[2]

    diagnostic_dict = read_diagnose(diagnosis_file)

    # Read file
    file = pandas.read_csv(fam_file, names=['FID', 'IID', 'father', 'mother', 'sex', 'phenotype'], index_col='IID',
                           delimiter=' ')
    headers = file.columns.tolist()

    out_file = open(pheno_file, 'w+')

    for key, data in file.iterrows():
        FID = data[headers.index('FID')]
        IID = key
        last_diagnose = diagnostic_dict[key]
        # '1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control
        if last_diagnose == 1 or last_diagnose == 2:
            # Mild cognitive impairment simply considered as not alzheimer
            out_file.write(f"{FID} {IID} {1}\n")
        elif last_diagnose == 3:
            out_file.write(f"{FID} {IID} {2}\n")
        else:
            out_file.write(f"{FID} {IID} {-9}\n")



        pass


if __name__ == "__main__":
    main(sys.argv[1:])
