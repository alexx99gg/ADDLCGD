import math

import dask
import numpy as np
import pandas

dask.config.set({"array.slicing.split_large_chunks": False})


def read_diagnose(file_path: str = '../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv', verbose=False):
    # Read diagnostic summary
    diagnostic_summary = pandas.read_csv(file_path, index_col='PTID')
    diagnostic_summary_headers = diagnostic_summary.columns.tolist()
    diagnostic_summary = diagnostic_summary.sort_values(by=["update_stamp"], ascending=True)
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


def generate_dataset(bed, bim, fam, snp_list):
    n_wgs_samples = bed.shape[1]
    n_snps = bed.shape[0]
    y = []
    # Generate label data
    # Keep only Alzheimer and cognitive normal patients (delete MCI patients)
    samples_to_keep = [True] * n_wgs_samples
    for i in range(n_wgs_samples):
        # Read phenotype data
        phenotype = int(fam.iat[i, 5])
        # print(phenotype)
        # According to plink, phenotype data values:
        # '1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control
        if phenotype == 1:
            # Cognitive normal
            label = 0
            y.append(label)
        elif phenotype == 2:
            label = 1
            y.append(label)
        else:
            # Remove this from the datasets as the diagnose is not clear
            samples_to_keep[i] = False
            n_wgs_samples -= 1
    y = np.asarray(y)
    y = y.reshape((n_wgs_samples,))

    # Generate features data

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

    # Filtering
    bed = bed[:, samples_to_keep]
    bed = bed[snps_to_keep, :]

    x = np.asarray(bed)
    x = x.transpose((1, 0))

    # Count NaN values
    n_NaN = np.count_nonzero(np.isnan(x))
    if n_NaN > 0:
        print(f"WARNING: number of missing genotypes in samples: {n_NaN}\n")
    # Change NaN values
    #   0 -> First allele
    #   1 -> Heterozygous
    #   2 -> Second allele
    #   math.nan -> missing genotype
    x = np.nan_to_num(x, 2)  # Change missing genotype to second allele (most common allele usually)
    # Normalize [0..1]
    n_final_snps = x.shape[1]
    if n_final_snps > 0:
        x = (x - np.min(x)) / np.ptp(x)
    # x = x[:, ~np.isnan(x).any(axis=0)] # Remove NaN values

    return x, y


def count_case_control(y):
    n_control = np.count_nonzero(y == 0)
    n_case = np.count_nonzero(y == 1)

    return n_control, n_case


def get_selected_snps(clumped_path):
    clump_file = pandas.read_csv(clumped_path, index_col='SNP', delimiter=r" +", engine='python')
    clump_headers = clump_file.columns.tolist()
    # Order by P value
    clump_file = clump_file.sort_values(by=["P"], ascending=True)
    snp_names = np.array(clump_file.index.tolist())
    snp_p_values = np.array(clump_file['P'].tolist())

    return snp_names, snp_p_values
