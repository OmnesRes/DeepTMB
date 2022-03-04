import pandas as pd
import pickle
import pyranges as pr
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]

##your path to the files directory
file_path = cwd / 'files'

usecols = ['Chromosome', 'Start_position', 'End_position', 'Variant_Classification', 'Variant_Type', 'Tumor_Sample_Barcode', 'Donor_ID']

##from: https://dcc.icgc.org/releases/PCAWG/consensus_snv_indel
pcawg_maf = pd.read_csv(file_path / 'final_consensus_passonly.snv_mnv_indel.icgc.public.maf', sep='\t',
                        usecols=usecols,
                        low_memory=False)


##from: https://dcc.icgc.org/releases/PCAWG/donors_and_biospecimens
pcawg_sample_table = pd.read_csv(file_path / 'pcawg_sample_sheet.tsv', sep='\t', low_memory=False)
##limit samples to what's in the maf
pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['aliquot_id'].isin(pcawg_maf['Tumor_Sample_Barcode'].unique())]
pcawg_sample_table.drop_duplicates(['icgc_donor_id'], inplace=True)
pcawg_sample_table = pcawg_sample_table.loc[pcawg_sample_table['dcc_specimen_type'] != 'Cell line - derived from tumour']
##from: https://dcc.icgc.org/releases/current/Summary
pcawg_donor_table = pd.read_csv(file_path / 'donor.all_projects.tsv', sep='\t', low_memory=False)
pcawg_sample_table = pd.merge(pcawg_sample_table, pcawg_donor_table, how='left', on='icgc_donor_id')

##limit MAF to unique samples
pcawg_maf = pcawg_maf.loc[pcawg_maf['Tumor_Sample_Barcode'].isin(pcawg_sample_table['aliquot_id'])]

# df of counts via groupby, could add other metrics derived from mc maf here
non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
pcawg_counts = pcawg_maf[['Variant_Classification', 'Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x), (x['Variant_Classification'].isin(non_syn)).sum()], index=['all_counts', 'non_syn_counts']))
pcawg_counts.reset_index(inplace=True)

# join to clinical annotation for data in mc3 only, this will add Tumor_Sample_Barcode also to the tcga_sample_table
pcawg_sample_table = pd.merge(pcawg_sample_table, pcawg_counts, how='right', left_on='aliquot_id', right_on='Tumor_Sample_Barcode')

##sample table is done, save to file
pickle.dump(pcawg_sample_table, open(file_path / 'pcawg_sample_table.pkl', 'wb'))

chromosomes = list(map(lambda x: str(x), list(range(1, 23)) + ['X', 'Y']))

##Use GFF3 to annotate variants
##ftp://ftp.ensembl.org/pub/grch37/current/gff3/homo_sapiens/
gff = pd.read_csv(file_path / 'Homo_sapiens.GRCh37.87.gff3',
                  sep='\t',
                  names=['chr', 'unknown', 'gene_part', 'start', 'end', 'unknown2', 'strand', 'unknown3', 'gene_info'],
                  usecols=['chr','gene_part', 'start', 'end', 'gene_info'],
                  low_memory=False)


gff_cds_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'CDS') & gff['chr'].isin(chromosomes), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge()
del gff

##make index column for merging
pcawg_maf['index'] = pcawg_maf.index.values

maf_pr = pr.PyRanges(pcawg_maf.loc[:, ['Chromosome', 'Start_position', 'End_position', 'index']].rename(columns={'Start_position': 'Start', 'End_position': 'End'}))

##10.1 synapse https://www.synapse.org/#!Synapse:syn21551261
genie = pd.read_csv(file_path / 'genie' / 'genomic_information.txt', sep='\t', low_memory=False)
panels = ['DFCI-ONCOPANEL-3', 'MDA-409-V1', 'MSK-IMPACT341', 'MSK-IMPACT468', 'VICC-01-R2']
panel_df = pd.DataFrame(data=panels, columns=['Panel'])
genie = genie.loc[genie['SEQ_ASSAY_ID'].isin(panels)]

total_sizes = []
cds_sizes = []
panel_prs = []

for panel in panels:
    print(panel)
    panel_pr = pr.PyRanges(genie.loc[(genie['SEQ_ASSAY_ID'] == panel) & genie['Chromosome'].isin(chromosomes), 'Chromosome':'End_Position'].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'})).merge()
    total_sizes.append(sum([i + 1 for i in panel_pr.lengths()]))
    cds_sizes.append(sum([i + 1 for i in panel_pr.intersect(gff_cds_pr).lengths()]))
    panel_prs.append(panel_pr)

grs = {k: v for k, v in zip(['CDS'] + list(panels), [gff_cds_pr] + panel_prs)}
result = pr.count_overlaps(grs, pr.concat({'maf': maf_pr}.values()))
result = result.df

pcawg_maf = pd.merge(pcawg_maf, result.iloc[:, 3:], how='left', on='index')

panel_df['total'] = total_sizes
panel_df['cds'] = cds_sizes

cds_total = sum([i + 1 for i in gff_cds_pr.lengths()])
panel_df = panel_df.append({'Panel': 'CDS', 'total': cds_total, 'cds': cds_total}, ignore_index=True)

pcawg_maf = pcawg_maf.loc[(pcawg_maf['DFCI-ONCOPANEL-3'] > 0) | (pcawg_maf['MDA-409-V1'] > 0) | (pcawg_maf['MSK-IMPACT341'] > 0) | (pcawg_maf['MSK-IMPACT468'] > 0) | (pcawg_maf['VICC-01-R2'] > 0)]

pcawg_maf.drop(columns=['index'], inplace=True)

pickle.dump(pcawg_maf, open(file_path / 'pcawg_maf_table.pkl', 'wb'))
pickle.dump(panel_df, open(file_path / 'pcawg_panel_table.pkl', 'wb'))