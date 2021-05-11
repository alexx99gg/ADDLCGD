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
        print_diagnostic_dict_summary(diagnostic_dict)
    return diagnostic_dict


def print_diagnostic_dict_summary(diagnostic_dict: dict):
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


def generate_dataset(diagnostic_dict: dict, bed, bim, fam, snp_list):
    n_wgs_samples = bed.shape[1]
    n_snps = bed.shape[0]
    y = []
    # Generate label data
    # Keep only Alzheimer and cognitive normal patients (delete MCI patients)
    samples_to_keep = [True] * n_wgs_samples
    i = 0
    for test in range(n_wgs_samples):
        # Read iid from wgs data
        iid = fam.iat[test, 1]
        # Get diagnose corresponding to the iid
        last_diagnose = diagnostic_dict[iid]
        has_alzheimer = -1
        if last_diagnose == 1:
            # Cognitive normal
            has_alzheimer = 0
            y.append(has_alzheimer)
        elif last_diagnose == 2:
            # Mild cognitive impairment
            # Remove this from the datasets as the diagnose is not clear
            samples_to_keep[i] = False
            n_wgs_samples -= 1
        elif last_diagnose == 3:
            has_alzheimer = 1
            y.append(has_alzheimer)
        else:
            print("Error: diagnosis not recognized")
            exit(1)
        i += 1
    y = np.asarray(y)
    y = y.reshape((n_wgs_samples, 1))

    # Generate features data
    x = np.asarray(bed)
    x = x.transpose((1, 0))
    # Delete MCI patients sample
    x = x[samples_to_keep, :]

    # Keep SNPs in snp_list only
    snps_to_keep = [False] * n_snps
    snps_not_found = 0
    for snp in snp_list:
        index = bim[bim['snp'] == snp].index
        if len(index) == 0:
            snps_not_found += 1
            continue
        index = index[0]
        snps_to_keep[index] = True
    if snps_not_found > 0:
        print(f"WARNING: SNPs from keep list not found: {snps_not_found}")
    x = x[:, snps_to_keep]
    # Count NaN values
    n_NaN = np.count_nonzero(np.isnan(x))
    if n_NaN > 0:
        print(f"WARNING: number of missing genotypes in samples: {n_NaN}\n")
    # Change NaN values
    #   0 -> First allele
    #   1 -> Heterozygous
    #   2 -> Second allele
    #   math.nan -> missing genotype
    x = np.nan_to_num(x, 1.5)
    # x = x[:, ~np.isnan(x).any(axis=0)] # Remove NaN values

    return x, y

def count_case_control(y):
    n_control = np.count_nonzero(y == 0)
    n_case = np.count_nonzero(y == 1)

    print(f"Number of control: {n_control}, number of cases: {n_case}\n")
    return n_control, n_case