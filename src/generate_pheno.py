import sys

import pandas

from read_data import *

def main(argv):
    # read params
    if len(argv) != 2:
        print("Usage: python3 generate_pheno.py fam_file diagnosis_file")
        exit(2)

    fam_file = argv[0]
    diagnosis_file = argv[1]

    diagnostic_dict = read_diagnose(diagnosis_file)

    # Read file
    file = pandas.read_csv(fam_file, names=['FID', 'IID', 'father', 'mother', 'sex', 'phenotype'], index_col='IID',
                           delimiter=' ')
    headers = file.columns.tolist()

    out_file = open(fam_file, 'w+')
    n_CN = 0
    n_MCI = 0
    n_AD = 0

    for key, data in file.iterrows():
        FID = data[headers.index('FID')]
        IID = key
        father = data[headers.index('father')]
        mother = data[headers.index('mother')]
        sex = data[headers.index('sex')]

        last_diagnose = diagnostic_dict[key]
        # '1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control
        if last_diagnose == 1:
            # Cognitive normal
            n_CN += 1
            out_file.write(f"{FID} {IID} {father} {mother} {sex} {1}\n")
        elif last_diagnose == 2:
            # Mild cognitive impairment
            n_MCI += 1
            out_file.write(f"{FID} {IID} {father} {mother} {sex} {-9}\n")
        elif last_diagnose == 3:
            n_AD += 1
            out_file.write(f"{FID} {IID} {father} {mother} {sex} {2}\n")
        else:
            print("ERROR: diagnose not recognized")
            exit(1)

    print("Phenotype data generated")
    print(f"Number of CN: {n_CN}")
    print(f"Number of MCI: {n_MCI}")
    print(f"Number of AD: {n_AD}")


if __name__ == "__main__":
    main(sys.argv[1:])
