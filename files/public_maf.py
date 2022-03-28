import pandas as pd
import numpy as np
import pyranges as pr
from tqdm import tqdm
import json
import pickle
import subprocess
import concurrent.futures

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
tcga_maf = tcga_maf.loc[~pd.isna(tcga_maf['Tumor_Seq_Allele2'])]
tcga_maf = tcga_maf.loc[~tcga_maf['Tumor_Seq_Allele2'].str.contains('N')]

tumor_to_normal = {}

for i in tcga_maf.itertuples():
    tumor_to_normal[i.Tumor_Sample_Barcode] = tumor_to_normal.get(i.Tumor_Sample_Barcode, []) + [i.Matched_Norm_Sample_Barcode]

for i in tumor_to_normal:
    tumor_to_normal[i] = set(tumor_to_normal[i])

##gdc data portal metadata files for TCGA WXS bams.  multiple files because only 10k can be added to the cart at a time.
with open(file_path / 'bams' / 'first_part.json', 'r') as f:
    metadata = json.load(f)

with open(file_path / 'bams' / 'second_part.json', 'r') as f:
    metadata += json.load(f)

with open(file_path / 'bams' / 'third_part.json', 'r') as f:
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


##there's a TNP that should only be merged into a DNP, remove the SNP then add it back
temp = tcga_maf.loc[(tcga_maf['Tumor_Sample_Barcode'] == 'TCGA-FW-A3R5-06A-11D-A23B-08') &\
                    (tcga_maf['Chromosome'] == '1') & (tcga_maf['Start_Position'] == 12725998)].copy()

tcga_maf = tcga_maf.loc[~((tcga_maf['Tumor_Sample_Barcode'] == 'TCGA-FW-A3R5-06A-11D-A23B-08') &\
                    (tcga_maf['Chromosome'] == '1') & (tcga_maf['Start_Position'] == 12725998))]


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
    if len(tumor_df) == 0:
        return None
    tumor_df.drop(columns=['bed', 'index'], inplace=True)
    tumor_df.sort_values(['Start_Position'], inplace=True)
    dfs = []
    for i in tumor_df['Chromosome'].unique():
        result = tumor_df.loc[(tumor_df['Chromosome'] == i) & (tumor_df['Variant_Type'] == 'SNP')].copy()
        if len(result) > 1:
            to_merge = sum(result['Start_Position'].values - result['Start_Position'].values[:, np.newaxis] == 1)
            merged = []
            position = 0
            indexes_to_remove = []
            while sum(to_merge[position:]) > 0 and position < len(to_merge) - 1:
                for index, merge in enumerate(to_merge[position:]):
                    if merge:
                        first = position + index - 1
                        last = position + index
                        while to_merge[last]:
                            last += 1
                            if last < len(to_merge):
                                pass
                            else:
                                break
                        position = last
                        last -= 1
                        variant = result.iloc[[first]].copy()
                        variant['End_Position'] = result.iloc[last]['Start_Position']
                        variant['Variant_Classification'] = 'Missense_Mutation'
                        if last - first == 1:
                            type = 'DNP'
                        elif last - first == 2:
                            type = 'TNP'
                        else:
                            type = 'ONP'
                        variant['Variant_Type'] = type
                        ref = ''
                        alt = ''
                        alt_counts = []
                        ref_counts = []
                        for row in result.iloc[first:last + 1, :].itertuples():
                            ref += row.Reference_Allele
                            alt += row.Tumor_Seq_Allele2
                            ref_counts.append(row.t_ref_count)
                            alt_counts.append(row.t_alt_count)
                        variant['t_ref_count'] = min(ref_counts)
                        variant['t_alt_count'] = min(alt_counts)
                        variant['Reference_Allele'] = ref
                        variant['Tumor_Seq_Allele2'] = alt
                        ##decide whether or not to merge
                        mean_vaf = np.mean([alt / (ref + alt) for ref, alt in zip(ref_counts, alt_counts)])
                        vaf_deviation = max([np.abs(mean_vaf - (alt / (ref + alt))) / mean_vaf for ref, alt in zip(ref_counts, alt_counts)])
                        ref_mean = max(np.mean(ref_counts), .00001)
                        ref_deviation_percent = max([np.abs(ref_mean - ref) / ref_mean for ref in ref_counts])
                        ref_deviation = max([np.abs(ref_mean - ref) for ref in ref_counts])
                        alt_mean = np.mean(alt_counts)
                        alt_deviation_percent = max([np.abs(alt_mean - alt) / alt_mean for alt in alt_counts])
                        alt_deviation = max([np.abs(alt_mean - alt) for alt in alt_counts])
                        if vaf_deviation < .05 or alt_deviation_percent < .05 or ref_deviation_percent < .05 or alt_deviation < 5 or ref_deviation < 5:
                            indexes_to_remove += list(range(first, last + 1))
                            merged.append(variant)
                        break
            result = result[~np.array([i in indexes_to_remove for i in range(len(result))])]
            if len(result) > 0 and len(merged) > 0:
                result = pd.concat([result, pd.concat(merged, ignore_index=True)], ignore_index=True)
            elif len(merged) == 0:
                pass
            else:
                result = pd.concat(merged, ignore_index=True)
            dfs.append(result)
        else:
            dfs.append(result)
    tumor_df = pd.concat([pd.concat(dfs, ignore_index=True), tumor_df.loc[tumor_df['Variant_Type'] != 'SNP'].copy()], ignore_index=True)
    return tumor_df

data = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for tumor, result in tqdm(zip(tumor_to_bed.keys(), executor.map(get_overlap, tumor_to_bed.keys()))):
        data[tumor] = result

final_df = pd.concat([data[i] for i in data if data[i] is not None] + [temp], ignore_index=True)

# load tcga clinical file  download from 'https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81'
tcga_sample_table = pd.read_csv(file_path / 'TCGA-CDR-SupplementalTableS1.tsv', sep='\t').iloc[:, 1:]
tcga_sample_table['histological_type'].fillna('', inplace=True)
tcga_sample_table = tcga_sample_table.loc[tcga_sample_table['bcr_patient_barcode'].isin(final_df['Tumor_Sample_Barcode'].str[:12].unique())]

patient_to_sample = {i[:12]: i[:16] for i in final_df['Tumor_Sample_Barcode'].unique()}
patient_to_barcode = {i[:12]: i for i in final_df['Tumor_Sample_Barcode'].unique()}

tcga_sample_table['bcr_sample_barcode'] = tcga_sample_table['bcr_patient_barcode'].apply(lambda x: patient_to_sample[x])
tcga_sample_table['Tumor_Sample_Barcode'] = tcga_sample_table['bcr_patient_barcode'].apply(lambda x: patient_to_barcode[x])

ncit = pd.read_csv(file_path / 'NCIt_labels.tsv', sep='\t')
ncit.fillna('', inplace=True)
ncit_labels_dict = {i.type + '_' + i.histological_type: i.NCIt_label for i in ncit.itertuples()}
ncit_codes_dict = {i.type + '_' + i.histological_type: i.NCIt_code for i in ncit.itertuples()}
ncit_tmb_labels_dict = {i.type + '_' + i.histological_type: i.NCIt_tmb_label for i in ncit.itertuples()}
ncit_tmb_codes_dict = {i.type + '_' + i.histological_type: i.NCIt_tmb_code for i in ncit.itertuples()}

ncit_labels = []
ncit_codes = []
ncit_tmb_labels = []
ncit_tmb_codes = []

for row in tcga_sample_table.itertuples():
    ncit_labels.append(ncit_labels_dict[row.type + '_' + row.histological_type])
    ncit_codes.append(ncit_codes_dict[row.type + '_' + row.histological_type])
    ncit_tmb_labels.append(ncit_tmb_labels_dict[row.type + '_' + row.histological_type])
    ncit_tmb_codes.append(ncit_tmb_codes_dict[row.type + '_' + row.histological_type])

tcga_sample_table['NCIt_label'] = ncit_labels
tcga_sample_table['NCIt_code'] = ncit_codes
tcga_sample_table['NCIt_tmb_label'] = ncit_tmb_labels
tcga_sample_table['NCIt_tmb_code'] = ncit_tmb_codes

with open(file_path / 'tcga_public_maf.pkl', 'wb') as f:
    pickle.dump(final_df, f)

with open(file_path / 'tcga_public_sample_table.pkl', 'wb') as f:
    pickle.dump(tcga_sample_table, f)