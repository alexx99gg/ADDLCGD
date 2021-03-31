import math

import numpy as np
import pandas


def read_diagnose(file_path: str = '../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv', verbose=False):
    # Read diagnostic summary
    diagnostic_summary = pandas.read_csv(file_path, index_col='PTID')
    diagnostic_summary_headers = diagnostic_summary.columns.tolist()

    # Create dictionary
    diagnostic_dict: dict = {}
    for key, data in diagnostic_summary.iterrows():
        # Iterate for each row of the document
        phase: str = data[diagnostic_summary_headers.index('Phase')]
        diagnosis: float = -1.
        if phase == "ADNI1":
            diagnosis = data[diagnostic_summary_headers.index('DXCURREN')]
        elif phase == "ADNI2" or phase == "ADNIGO":
            dxchange = data[diagnostic_summary_headers.index('DXCHANGE')]
            if dxchange == 1 or dxchange == 7 or dxchange == 9:
                diagnosis = 1.
            if dxchange == 2 or dxchange == 4 or dxchange == 8:
                diagnosis = 2.
            if dxchange == 3 or dxchange == 5 or dxchange == 6:
                diagnosis = 3.
        elif phase == "ADNI3":
            diagnosis = data[diagnostic_summary_headers.index('DIAGNOSIS')]
        else:
            print(f"ERROR: Not recognized study phase {phase}")
            exit(1)
        # Update dictionary
        if not math.isnan(diagnosis):
            diagnostic_dict[key] = diagnosis
    if verbose:
        print_diagnostic_summary(diagnostic_dict)
    return diagnostic_dict


def print_diagnostic_summary(diagnostic_dict: dict):
    print(f"Number of diagnosed patients: {len(diagnostic_dict.items())}\n")
    n_NL = 0
    n_MCI = 0
    n_AD = 0
    for (key, data) in diagnostic_dict.items():
        if data == 1:
            n_NL += 1
        if data == 2:
            n_MCI += 1
        if data == 3:
            n_AD += 1
    print(f"Number of NL patients: {n_NL}\n"
          f"Number of MCI patients: {n_MCI}\n"
          f"Number of AD patients: {n_AD}\n")


def generate_dataset(diagnostic_dict: dict, fam, bed):
    n_wgs_samples = bed.shape[1]
    n_snps = bed.shape[0]
    y = []
    # Generate label data
    # Only difference between Alzheimer and not alzheimer
    for test in range(n_wgs_samples):
        # Read iid from wgs data
        iid = fam.iat[test, 1]
        # Get diagnose corresponding to the iid
        last_diagnose = diagnostic_dict[iid]
        has_alzheimer = False
        if last_diagnose == 2 or last_diagnose == 1:
            # Mild cognitive impairment simply considered not alzheimer
            has_alzheimer = False
        elif last_diagnose == 3:
            has_alzheimer = True
        else:
            print("Error: diagnosis not recognized")
        y.append(has_alzheimer)
    y = np.asarray(y)
    y = y.reshape((n_wgs_samples, 1))

    # Generate features data
    x = np.asarray(bed)
    x = x.transpose((1, 0))

    # Normalize dataset values [0,1]
    x = (x - np.min(x)) / np.ptp(x)
    return x, y
