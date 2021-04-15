import pandas


def main(file_path: str = '../IGAP_data/IGAP_stage_1_2_combined.txt', out_file='../IGAP_data/IGAP.assoc'):
    # Read file
    file = pandas.read_csv(file_path, index_col='MarkerName', delimiter='	')
    headers = file.columns.tolist()

    out_file = open(out_file, 'w+')

    out_file.write(f"CHR\tSNP\tBP\tA1\tA2\tBETA\tSE\tP\n")
    for key, data in file.iterrows():
        CHR = data[headers.index('Chromosome')]
        SNP = key
        BP = data[headers.index('Position')]
        A1 = data[headers.index('Effect_allele')]
        A2 = data[headers.index('Non_Effect_allele')]
        BETA = data[headers.index('Beta')]
        SE = data[headers.index('SE')]
        P = data[headers.index('Pvalue')]

        out_file.write(f"{CHR}\t{SNP}\t{BP}\t{A1}\t{A2}\t{BETA}\t{SE}\t{P}\n")


if __name__ == "__main__":
    main()
