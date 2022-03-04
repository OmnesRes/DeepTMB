import numpy as np
import pandas as pd
file_path = 'files/'
import pickle
import concurrent.futures
import pyranges as pr
import json
import subprocess
from tqdm import tqdm

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

depths = pickle.load(open(cwd / 'files' / 'depths.pkl', 'rb'))

with open(cwd / 'files' / 'bams' / 'first_part.json', 'r') as f:
    metadata = json.load(f)

with open(cwd / 'files' / 'bams' / 'second_part.json', 'r') as f:
    metadata += json.load(f)

with open(cwd / 'files' / 'bams' / 'third_part.json', 'r') as f:
    metadata += json.load(f)

sample_to_id = {}
for i in metadata:
    sample_to_id[i['associated_entities'][0]['entity_submitter_id']] = i['associated_entities'][0]['entity_id']

cmd = ['ls', cwd / 'files' / 'beds']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
files = [str(i, 'utf-8') for i in p.communicate()[0].split() if '.bed' in str(i)[-5:]]

tcga_maf = pickle.load(open(cwd / 'files' / 'tcga_public_maf.pkl', 'rb'))

tumor_to_normal = tcga_maf[['Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode']].groupby('Tumor_Sample_Barcode')['Matched_Norm_Sample_Barcode'].apply(lambda x: set(x.values)).to_dict()

tumor_to_bed = {}
for i in tumor_to_normal:
    if i in sample_to_id and list(tumor_to_normal[i])[0] in sample_to_id:
        for j in files:
            if sample_to_id[i] in j and sample_to_id[list(tumor_to_normal[i])[0]] in j:
                tumor_to_bed[i] = j

chromosomes = list(map(lambda x: str(x), list(range(1, 23)) + ['X', 'Y']))

##need to intersect coverage files with panel coordinates
##10.1 synapse https://www.synapse.org/#!Synapse:syn25895958
genie = pd.read_csv(cwd / 'files' / 'genie' / 'genomic_information.txt', sep='\t', low_memory=False)
panel_pr = pr.PyRanges(genie.loc[genie['SEQ_ASSAY_ID'] == 'MSK-IMPACT341'][['Chromosome', 'Start_Position', 'End_Position']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))

non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
gff = pd.read_csv(cwd / 'files' / 'Homo_sapiens.GRCh37.87.gff3',
                  sep='\t',
                  names=['chr', 'unknown', 'gene_part', 'start', 'end', 'unknown2', 'strand', 'unknown3', 'gene_info'],
                  usecols=['chr', 'gene_part', 'start', 'end', 'gene_info'],
                  low_memory=False)
gff_cds_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'CDS') & gff['chr'].isin(chromosomes), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge()

def get_overlap(tumor):
    tumor_df = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'] == tumor]
    counts = depths[tumor][-1]
    if len(counts) == 0:
        return None
    if np.median(counts[counts > 0]) < 30:
        return None
    file = tumor_to_bed[tumor]
    try:
        bed_df = pd.read_csv(cwd / 'files' / 'beds' / file, names=['Chromosome', 'Start', 'End'], low_memory=False, sep='\t')
    except:
        return None
    bed_df = bed_df.loc[bed_df['Chromosome'].isin(chromosomes)]
    bed_pr = pr.PyRanges(bed_df).merge()
    bed_cds_pr = bed_pr.intersect(gff_cds_pr).merge()
    bed_size = sum([i + 1 for i in bed_cds_pr.lengths()])
    bed_panel_cds_pr = bed_cds_pr.intersect(panel_pr).merge()
    panel_size = sum([i + 1 for i in bed_panel_cds_pr.lengths()])
    tumor_pr = pr.PyRanges(tumor_df[['Chromosome', 'Start_Position', 'End_Position', 'Variant_Classification', 'Variant_Type']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))
    grs = {'bed_cds': bed_cds_pr, 'bed_panel_cds': bed_panel_cds_pr}
    result = pr.count_overlaps(grs, pr.concat({'maf': tumor_pr}.values()))
    result = result.df
    exome_counts = sum((result['bed_cds'] > 0) & result['Variant_Classification'].isin(non_syn))
    if exome_counts == 0:
        return None
    if bed_size < 25000000:
        return None
    panel_counts = sum(result['bed_panel_cds'] > 0)
    types = result[result['bed_panel_cds'] > 0]['Variant_Classification'].value_counts()
    classes = result[result['bed_panel_cds'] > 0]['Variant_Type'].value_counts()
    return [panel_counts, panel_size, exome_counts, bed_size, types, classes]

data = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    for tumor, result in tqdm(zip(tumor_to_bed.keys(), executor.map(get_overlap, tumor_to_bed.keys()))):
        data[tumor] = result

with open(cwd / 'tables' / 'table_1' / 'MSK_341' / 'data' / 'data.pkl', 'wb') as f:
    pickle.dump(data, f)
