import pandas
from pandas_plink import read_plink1_bin

# Read diagnostic summary
diagnostic_summary = pandas.read_csv('../diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv', index_col='PTID')
diagnostic_summary_headers = diagnostic_summary.columns.tolist()

# Create dictionary
diagnostic_dict: dict = {}
n_NL = 0
n_MCI = 0
n_AD = 0
for key, data in diagnostic_summary.iterrows():
    phase: str = data[diagnostic_summary_headers.index('Phase')]
    diagnosis: float = -1.
    """
    ADNI1:
        Column: DXCURREN
            1=NL;
            2=MCI;
            3=AD
    ADNIGO/2:
        Column: DXCHANGE
            1=Stable: NL to NL;
            2=Stable: MCI to MCI;
            3=Stable: Dementia to Dementia;
            4=Conversion: NL to MCI;
            5=Conversion: MCI to Dementia;
            6=Conversion: NL to Dementia;
            7=Reversion: MCI to NL;
            8=Reversion: Dementia to MCI;
            9=Reversion: Dementia to NL
    ADNI3:
        Column: DIAGNOSIS
            1=CN;
            2=MCI;
            3=Dementia
    """
    if phase == "ADNI1":
        diagnosis = data[diagnostic_summary_headers.index('DXCURREN')]
    if phase == "ADNI2" or phase == "ADNIGO":
        dxchange = data[diagnostic_summary_headers.index('DXCHANGE')]
        if dxchange == 1 or dxchange == 7 or dxchange == 9:
            diagnosis = 1.
        if dxchange == 2 or dxchange == 4 or dxchange == 8:
            diagnosis = 2.
        if dxchange == 3 or dxchange == 5 or dxchange == 6:
            diagnosis = 3.
    if phase == "ADNI3":
        diagnosis = data[diagnostic_summary_headers.index('DIAGNOSIS')]
    # Update dictionary
    diagnostic_dict[key] = diagnosis

print(f"Number of diagnosed patients: {len(diagnostic_dict.items())}\n")
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

# Read WGS data
WGS_ADNI_1 = read_plink1_bin("../wgs_data/ADNI_cluster_01_forward_757LONI.bed",
                             "../wgs_data/ADNI_cluster_01_forward_757LONI.bim",
                             "../wgs_data/ADNI_cluster_01_forward_757LONI.fam", verbose=False)
WGS_ADNI_GO_2_Forward = read_plink1_bin("../wgs_data/ADNI_GO_2_Forward_Bin.bed",
                                        "../wgs_data/ADNI_GO_2_Forward_Bin.bim",
                                        "../wgs_data/ADNI_GO_2_Forward_Bin.fam", verbose=False)
WGS_ADNI_GO_2_2nd_orig = read_plink1_bin("../wgs_data/ADNI_GO2_GWAS_2nd_orig_BIN.bed",
                                        "../wgs_data/ADNI_GO2_GWAS_2nd_orig_BIN.bim",
                                        "../wgs_data/ADNI_GO2_GWAS_2nd_orig_BIN.fam", verbose=False)
WGS_ADNI_3 = read_plink1_bin("../wgs_data/ADNI3_PLINK_Final.bed",
                             "../wgs_data/ADNI3_PLINK_Final.bim",
                             "../wgs_data/ADNI3_PLINK_Final.fam", verbose=False)

index_ADNI_1 = WGS_ADNI_1.indexes
index_ADNI_GO_2_Forward = WGS_ADNI_GO_2_Forward.indexes
index_ADNI_GO_2_2nd_orig = WGS_ADNI_GO_2_2nd_orig.indexes
index_ADNI_3 = WGS_ADNI_3.indexes

# TEMP: check for id matches (shouldn't occur)
for i in index_ADNI_1.get('sample'):
    for j in index_ADNI_GO_2_Forward.get('sample'):
        if i == j:
            print("MATCH")
    for j in index_ADNI_GO_2_2nd_orig.get('sample'):
        if i == j:
            print("MATCH")
    for j in index_ADNI_3.get('sample'):
        if i == j:
            print("MATCH")

for i in index_ADNI_GO_2_Forward.get('sample'):
    for j in index_ADNI_GO_2_2nd_orig.get('sample'):
        if i == j:
            print("MATCH")
    for j in index_ADNI_3.get('sample'):
        if i == j:
            print("MATCH")

print(f"Current number of WGS samples in ADNI_1: {len(index_ADNI_1.get('sample'))}")
print(f"Current number of WGS samples in ADNI_2_Forward: {len(index_ADNI_GO_2_Forward.get('sample'))}")
print(f"Current number of WGS samples in ADNI_2_2nd_orig: {len(index_ADNI_GO_2_2nd_orig.get('sample'))}")
print(f"Current number of WGS samples in ADNI_3: {len(index_ADNI_3.get('sample'))}\n")

print(f"Current number of variants per WGS sample: {len(WGS_ADNI_1.indexes.get('variant'))}\n")
print(f"Current number of variants per WGS sample: {len(WGS_ADNI_3.indexes.get('variant'))}\n")

# TODO: get SNPs
