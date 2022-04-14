##the annotation file should already be restricted to the CDS and pancan GENIE
##contains somatic and germline
##somatic contains some merged mutations, need to unmerge and then merge with germline
##some duplicates between somatic and germline, need to drop the correct rows

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

annot, maf = pickle.load(open(file_path / 'germline' / 'data' / 'tcga.genie.combined.annot.maf.pkl', 'rb'))

maf['bcr_patient_barcode'] = maf['SAMPLE'].apply(lambda x: x[:12])

usecols = ['Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode']

tcga_maf = pd.read_csv(file_path / 'mc3.v0.2.8.PUBLIC.maf', sep='\t', usecols=usecols, low_memory=False).drop_duplicates()

tumor_to_normal = {}

for i in tcga_maf.itertuples():
    tumor_to_normal[i.Tumor_Sample_Barcode] = tumor_to_normal.get(i.Tumor_Sample_Barcode, []) + [i.Matched_Norm_Sample_Barcode]

del tcga_maf
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

def get_overlap(tumor):
    file = tumor_to_bed[tumor]
    try:
        bed_df = pd.read_csv(file_path / 'beds' / file, names=['Chromosome', 'Start', 'End'], low_memory=False, sep='\t')
    except:
        return None
    bed_df = bed_df.loc[bed_df['Chromosome'].isin(chromosomes)]
    bed_pr = pr.PyRanges(bed_df).merge()
    tumor_df = maf.loc[maf['bcr_patient_barcode'] == tumor[:12]]
    if len(tumor_df) == 0:
        return None
    tumor_df['index'] = tumor_df.index.values
    tumor_df['End_Position'] = tumor_df['POS'] + tumor_df['REF_ALLELE'].str.len() - 1
    tumor_df.loc[(tumor_df['REF_ALLELE'].str.len() == 1) & (tumor_df['ALT_ALLELE'].str.len() > tumor_df['REF_ALLELE'].str.len()), ['End_Position']] = tumor_df['POS'] + 1
    tumor_pr = pr.PyRanges(tumor_df[['CHROM', 'POS', 'End_Position', 'index']].rename(columns={'CHROM': 'Chromosome', 'POS': 'Start', 'End_Position': 'End'}))
    grs = {'bed': bed_pr}
    result = pr.count_overlaps(grs, pr.concat({'maf': tumor_pr}.values()))
    result = result.df
    tumor_df = pd.merge(tumor_df, result.iloc[:, 3:], how='left', on='index')
    tumor_df = tumor_df.loc[tumor_df['bed'] > 0]
    if len(tumor_df) == 0:
        return None
    tumor_df.drop(columns=['bed', 'index'], inplace=True)
    germline_df = tumor_df[tumor_df['LINEAGE'] == 'germline']
    germline_df.reset_index(inplace=True, drop=True)
    somatic_df = tumor_df[tumor_df['LINEAGE'] == 'somatic']
    ##need to split into single substitutions since that is how germline is reported
    MNPs = somatic_df.loc[(somatic_df['REF_ALLELE'].str.len() > 1) &
                          (somatic_df['ALT_ALLELE'].str.len() > 1) &
                          (somatic_df['REF_ALLELE'].str.len() == somatic_df['ALT_ALLELE'].str.len())].copy()

    somatic_df = somatic_df.loc[~((somatic_df['REF_ALLELE'].str.len() > 1) &
                                  (somatic_df['ALT_ALLELE'].str.len() > 1) &
                                  (somatic_df['REF_ALLELE'].str.len() == somatic_df['ALT_ALLELE'].str.len()))]
    variants = []
    for index in range(len(MNPs)):
        variant = MNPs.iloc[[index]].copy()
        for position, (ref, alt) in enumerate(zip(variant['REF_ALLELE'].str, variant['ALT_ALLELE'].str)):
            new_variant = variant.copy()
            new_variant['REF_ALLELE'] = ref
            new_variant['ALT_ALLELE'] = alt
            new_variant['POS'] = variant['POS'].values[0] + position
            variants.append(new_variant)
    if len(variants) == 0:
        pass
    else:
        new_SNPs = pd.concat(variants, ignore_index=True)
        somatic_df = pd.concat([somatic_df, new_SNPs], ignore_index=True)
    germline_df['germline_index'] = germline_df.index.values
    somatic_df['somatic_index'] = somatic_df.index.values
    merged = pd.merge(germline_df[['CHROM', 'POS', 'REF_ALLELE', 'ALT_ALLELE', 'SAMPLE', 'germline_index']], somatic_df[['CHROM', 'POS', 'REF_ALLELE', 'ALT_ALLELE', 'SAMPLE', 'somatic_index']], on=['CHROM', 'POS', 'REF_ALLELE', 'ALT_ALLELE'], how='inner')
    somatic_indexes = merged.loc[merged['SAMPLE_x'].str[13:15].astype(np.int32) >= 10]['somatic_index'].values
    germline_indexes = merged.loc[merged['SAMPLE_x'].str[13:15].astype(np.int32) < 10]['germline_index'].values
    if len(merged) > 0:
        somatic_df = somatic_df[~np.array([i in somatic_indexes for i in range(len(somatic_df))])]
        germline_df = germline_df[~np.array([i in germline_indexes for i in range(len(germline_df))])]
    somatic_df.drop(columns=['somatic_index'], inplace=True)
    germline_df.drop(columns=['germline_index'], inplace=True)
    tumor_df = pd.concat([somatic_df, germline_df], ignore_index=True)
    tumor_df.sort_values(['POS'], inplace=True)
    dfs = []
    for i in tumor_df['CHROM'].unique():
        result = tumor_df.loc[(tumor_df['CHROM'] == i) & (tumor_df['REF_ALLELE'].str.len() == tumor_df['ALT_ALLELE'].str.len())].copy()
        if len(result) > 1:
            to_merge = sum(result['POS'].values - result['POS'].values[:, np.newaxis] == 1)
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
                        ref = ''
                        alt = ''
                        alt_counts = []
                        ref_counts = []
                        origins = []
                        for row in result.iloc[first:last + 1, :].itertuples():
                            ref += row.REF_ALLELE
                            alt += row.ALT_ALLELE
                            ref_counts.append(row.REF_DEPTH)
                            alt_counts.append(row.ALT_DEPTH)
                            origins.append(row.LINEAGE)
                        variant['REF_DEPTH'] = min(ref_counts)
                        variant['ALT_DEPTH'] = min(alt_counts)
                        variant['REF_ALLELE'] = ref
                        variant['ALT_ALLELE'] = alt
                        if len(set(origins)) == 1:
                            variant['LINEAGE'] = list(origins)[0]
                        else:
                            variant['LINEAGE'] = 'both'
                        if np.isnan(ref_counts).any() or np.isnan(alt_counts).any():
                            pass
                        else:
                            ##decide whether or not to merge
                            mean_vaf = np.mean([alt / (ref + alt) for ref, alt in zip(ref_counts, alt_counts)])
                            vaf_deviation = max([np.abs(mean_vaf - (alt / (ref + alt))) / mean_vaf for ref, alt in zip(ref_counts, alt_counts)])
                            ref_mean = max(np.mean(ref_counts), .00001)
                            ref_deviation_percent = max([np.abs(ref_mean - ref) / ref_mean for ref in ref_counts])
                            ref_deviation = max([np.abs(ref_mean - ref) for ref in ref_counts])
                            alt_mean = np.mean(alt_counts)
                            alt_deviation_percent = max([np.abs(alt_mean - alt) / alt_mean for alt in alt_counts])
                            alt_deviation = max([np.abs(alt_mean - alt) for alt in alt_counts])
                            if len(set(origins)) == 1:
                                if vaf_deviation < .05 or alt_deviation_percent < .05 or ref_deviation_percent < .05 or alt_deviation < 5 or ref_deviation < 5:
                                    indexes_to_remove += list(range(first, last + 1))
                                    merged.append(variant)
                            else:
                                if vaf_deviation < .1:
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
    tumor_df = pd.concat([pd.concat(dfs, ignore_index=True), tumor_df.loc[(tumor_df['REF_ALLELE'].str.len() != tumor_df['ALT_ALLELE'].str.len())].copy()], ignore_index=True)
    return tumor_df

data = {}
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for tumor, result in tqdm(zip(tumor_to_bed.keys(), executor.map(get_overlap, tumor_to_bed.keys()))):
        data[tumor] = result

with open(file_path / 'germline' / 'data' / 'tumor_only_maf.pkl', 'wb') as f:
    pickle.dump(maf, f)