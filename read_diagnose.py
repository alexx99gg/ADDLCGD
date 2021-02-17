import pandas
from pandas_plink import read_plink1_bin

# Read diagnostic summary
diagnostic_summary = pandas.read_csv('diagnosis_data/DXSUM_PDXCONV_ADNIALL.csv', index_col='PTID')
diagnostic_summary_headers = diagnostic_summary.columns.tolist()

# Create dictionary
diagnostic_dict: dict = {}
for key, data in diagnostic_summary.iterrows():
    phase: str = data[diagnostic_summary_headers.index('Phase')]
    diagnosis: float = -1.
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

print(len(diagnostic_dict))
print(diagnostic_dict)

# Read WGS data
WGS_ADNI_1 = read_plink1_bin("wgs_data/ADNI_cluster_01_forward_757LONI.bed",
                             "wgs_data/ADNI_cluster_01_forward_757LONI.bim",
                             "wgs_data/ADNI_cluster_01_forward_757LONI.fam", verbose=False)
WGS_ADNI_GO_2 = read_plink1_bin("wgs_data/ADNI_GO_2_Forward_Bin.bed",
                                "wgs_data/ADNI_GO_2_Forward_Bin.bim",
                                "wgs_data/ADNI_GO_2_Forward_Bin.fam", verbose=False)
WGS_ADNI_3 = read_plink1_bin("wgs_data/ADNI3_PLINK_Final.bed",
                            "wgs_data/ADNI3_PLINK_Final.bim",
                            "wgs_data/ADNI3_PLINK_Final.fam", verbose=False)

index_ADNI_1 = WGS_ADNI_1.indexes
index_ADNI_GO_2 = WGS_ADNI_GO_2.indexes
index_ADNI_3 = WGS_ADNI_3.indexes

# TEMP: check for id matches
for i in index_ADNI_1.get('sample'):
    for j in index_ADNI_GO_2.get('sample'):
        if i == j:
            print("MATCH")
    for j in index_ADNI_3.get('sample'):
        if i == j:
            print("MATCH")

for i in index_ADNI_GO_2.get('sample'):
    for j in index_ADNI_3.get('sample'):
        if i == j:
            print("MATCH")
