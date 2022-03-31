import pandas as pd
import pyranges as pr
import pickle

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

##your path to the files directory
file_path = cwd / 'files'

somatic_maf = pickle.load(open(file_path / 'tcga_public_maf.pkl', 'rb'))
somatic_maf = somatic_maf.loc[:, ['Chromosome', 'Start_Position', 'End_Position',
       'Variant_Classification', 'Variant_Type', 'Reference_Allele',
       'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode',
       'Matched_Norm_Sample_Barcode', 't_ref_count', 't_alt_count', 'STRAND',
       'FILTER']]

gff = pd.read_csv(file_path / 'Homo_sapiens.GRCh37.87.gff3',
                  sep='\t',
                  names=['chr', 'source', 'gene_part', 'start', 'end', 'unknown', 'strand', 'unknown2', 'gene_info'],
                  usecols=['chr', 'source', 'gene_part', 'start', 'end', 'gene_info'],
                  low_memory=False)

chromosomes = list(map(lambda x: str(x), list(range(1, 23)) + ['X', 'Y']))
gff_cds_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'CDS') & gff['chr'].isin(chromosomes), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge()

genie = pd.read_csv(cwd / 'files' / 'genomic_information.txt', sep='\t', low_memory=False)
genie_pr = pr.PyRanges(genie[['Chromosome', 'Start_Position', 'End_Position']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))
genie_cds_pr = genie_pr.intersect(gff_cds_pr).merge()

##limit the MAF
grs = {'genie_cds': genie_cds_pr}
somatic_maf['index'] = somatic_maf.index.values
somatic_pr = pr.PyRanges(somatic_maf[['Chromosome', 'Start_Position', 'End_Position', 'index']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))
result = pr.count_overlaps(grs, pr.concat({'maf': somatic_pr}.values()))
result = result.df
somatic_maf = pd.merge(somatic_maf, result.iloc[:, 3:], how='left', on='index')
somatic_maf = somatic_maf.loc[somatic_maf['genie_cds'] > 0]
somatic_maf.drop(columns=['genie_cds', 'index'], inplace=True)

##need to split into single substitutions since that is how germline is reported
MNPs = somatic_maf.loc[(somatic_maf['Reference_Allele'].str.len() > 1) &\
                       (somatic_maf['Tumor_Seq_Allele2'].str.len() > 1) &\
                       (somatic_maf['Reference_Allele'].str.len() == somatic_maf['Tumor_Seq_Allele2'].str.len())].copy()

somatic_maf = somatic_maf.loc[~((somatic_maf['Reference_Allele'].str.len() > 1) &\
                       (somatic_maf['Tumor_Seq_Allele2'].str.len() > 1) &\
                       (somatic_maf['Reference_Allele'].str.len() == somatic_maf['Tumor_Seq_Allele2'].str.len()))]

variants = []
for index in range(len(MNPs)):
    variant = MNPs.iloc[[index]].copy()
    for position, (ref, alt) in enumerate(zip(variant['Reference_Allele'].str, variant['Tumor_Seq_Allele2'].str)):
        new_variant = variant.copy()
        new_variant['Reference_Allele'] = ref
        new_variant['Tumor_Seq_Allele2'] = alt
        new_variant['Start_Position'] = variant['Start_Position'].values[0] + position
        new_variant['End_Position'] = variant['Start_Position'].values[0] + position
        new_variant['Variant_Type'] = 'SNP'
        variants.append(new_variant)

new_SNPs = pd.concat(variants, ignore_index=True)

somatic_maf = pd.concat([somatic_maf, new_SNPs], ignore_index=True)

somatic_maf.to_csv(file_path / 'germline' / 'data' / 'pan-genie-cds-mc3.maf', sep='\t', index=False)