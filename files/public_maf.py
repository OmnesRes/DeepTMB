import pandas as pd
import pyranges as pr
from tqdm import tqdm
import json
import pickle
import subprocess

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

usecols = ['Chromosome', 'Start_Position', 'End_Position', 'STRAND', 'Variant_Type', 'Variant_Classification', 'Reference_Allele', 'Tumor_Seq_Allele2', 't_ref_count', 't_alt_count', 'Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode', 'FILTER']

tcga_maf = pd.read_csv(file_path / 'mc3.v0.2.8.PUBLIC.maf', sep='\t', usecols=usecols, low_memory=False)

##The MAF contains nonpreferred pairs which results in some samples having duplicated variants
tcga_maf = tcga_maf.loc[(tcga_maf['FILTER'] == 'PASS') | (tcga_maf['FILTER'] == 'wga') | (tcga_maf['FILTER'] == 'native_wga_mix')]

tumor_to_normal = {}

for i in tcga_maf.itertuples():
    tumor_to_normal[i.Tumor_Sample_Barcode] = tumor_to_normal.get(i.Tumor_Sample_Barcode, []) + [i.Matched_Norm_Sample_Barcode]

for i in tumor_to_normal:
    tumor_to_normal[i] = set(tumor_to_normal[i])

##gdc data portal metadata files for TCGA WXS bams.  multiple files because only 10k can be added to the cart at a time.
with open(file_path / '/bams/first_part.json', 'r') as f:
    metadata = json.load(f)

with open(file_path / '/bams/second_part.json', 'r') as f:
    metadata += json.load(f)

with open(file_path / '/bams/third_part.json', 'r') as f:
    metadata += json.load(f)

sample_to_id = {}
for i in metadata:
    sample_to_id[i['associated_entities'][0]['entity_submitter_id']] = i['associated_entities'][0]['entity_id']

cmd = ['ls', 'files/beds']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
files = [str(i, 'utf-8') for i in p.communicate()[0].split() if '.bed' in str(i)[-5:]]

tumor_to_bed = {}
for i in tumor_to_normal:
    if i in sample_to_id and list(tumor_to_normal[i])[0] in sample_to_id:
        for j in files:
            if sample_to_id[i] in j and sample_to_id[list(tumor_to_normal[i])[0]] in j:
                tumor_to_bed[i] = j


chromosomes = list(map(lambda x: str(x), list(range(1, 23)) + ['X', 'Y']))

import concurrent.futures

def get_overlap(tumor):
    file = tumor_to_bed[tumor]
    try:
        bed_df = pd.read_csv(file_path / 'beds' / file, names=['Chromosome', 'Start', 'End'], low_memory=False, sep='\t')
    except:
        return None
    bed_df = bed_df.loc[bed_df['Chromosome'].isin(chromosomes)]
    bed_pr = pr.PyRanges(bed_df).merge()
    tumor_df = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'] == tumor]
    tumor_df['index'] = tumor_df.index.values
    tumor_pr = pr.PyRanges(tumor_df[['Chromosome', 'Start_Position', 'End_Position', 'index']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))
    grs = {'bed': bed_pr}
    result = pr.count_overlaps(grs, pr.concat({'maf': tumor_pr}.values()))
    result = result.df
    tumor_df = pd.merge(tumor_df, result.iloc[:, 3:], how='left', on='index')
    tumor_df = tumor_df.loc[tumor_df['bed'] > 0]
    tumor_df.drop(columns=['bed', 'index'], inplace=True)
    return tumor_df

data = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    for tumor, result in tqdm(zip(tumor_to_bed.keys(), executor.map(get_overlap, tumor_to_bed.keys()))):
        data[tumor] = result

final_df = pd.concat([data[i] for i in data if data[i] is not None], ignore_index=True)

# load tcga clinical file  download from 'https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81'
tcga_sample_table = pd.read_csv(file_path / 'TCGA-CDR-SupplementalTableS1.tsv', sep='\t').iloc[:, 1:]
tcga_sample_table['histological_type'].fillna('', inplace=True)
tcga_sample_table = tcga_sample_table.loc[tcga_sample_table['bcr_patient_barcode'].isin(final_df['Tumor_Sample_Barcode'].str[:12].unique())]

patient_to_sample = {i[:12]: i[:16] for i in final_df['Tumor_Sample_Barcode'].unique()}
patient_to_barcode = {i[:12]: i for i in final_df['Tumor_Sample_Barcode'].unique()}

tcga_sample_table['bcr_sample_barcode'] = tcga_sample_table['bcr_patient_barcode'].apply(lambda x: patient_to_sample[x])
tcga_sample_table['Tumor_Sample_Barcode'] = tcga_sample_table['bcr_patient_barcode'].apply(lambda x: patient_to_barcode[x])

with open(file_path / 'tcga_public_maf.pkl', 'wb') as f:
    pickle.dump(final_df, f)

with open(file_path / 'tcga_public_sample_table.pkl', 'wb') as f:
    pickle.dump(tcga_sample_table, f)
