import pickle
import pandas as pd
import numpy as np

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

file_path = cwd / 'files'

annot, starting_maf = pickle.load(open(file_path / 'germline' / 'data' / 'tcga.genie.combined.annot.maf.pkl', 'rb'))
del starting_maf

maf = pickle.load(open(file_path / 'germline' / 'data' / 'tumor_only_maf.pkl', 'rb'))
maf = maf[['LINEAGE', 'CHROM', 'POS', 'REF_ALLELE', 'ALT_ALLELE', 'ID']]
t_s = sum(maf['LINEAGE'] == 'somatic') + sum(maf['LINEAGE'] == 'both')
t_g = sum(maf['LINEAGE'] == 'germline') + sum(maf['LINEAGE'] == 'both')

merged = maf.loc[(maf['REF_ALLELE'].str.len() > 1) & (maf['ALT_ALLELE'].str.len() > 1) & (maf['REF_ALLELE'].str.len() == maf['ALT_ALLELE'].str.len())]
unmerged = maf.loc[~((maf['REF_ALLELE'].str.len() > 1) & (maf['ALT_ALLELE'].str.len() > 1) & (maf['REF_ALLELE'].str.len() == maf['ALT_ALLELE'].str.len()))]

germline_counts = unmerged.loc[unmerged['LINEAGE'] == 'germline']['ID'].value_counts().to_dict()
unmerged['germline_count'] = unmerged['ID'].apply(lambda x: germline_counts.get(x, 0))
somatic_counts = unmerged.loc[unmerged['LINEAGE'] == 'somatic']['ID'].value_counts().to_dict()
unmerged['somatic_count'] = unmerged['ID'].apply(lambda x: somatic_counts.get(x, 0))

merged['merged_ID'] = merged['ID'] + '_' + merged['REF_ALLELE']
germline_counts = merged.loc[merged['LINEAGE'].isin(['germline', 'both'])]['merged_ID'].value_counts().to_dict()
merged['germline_count'] = merged['merged_ID'].apply(lambda x: germline_counts.get(x, 0))
somatic_counts = merged.loc[merged['LINEAGE'].isin(['somatic', 'both'])]['merged_ID'].value_counts().to_dict()
merged['somatic_count'] = merged['merged_ID'].apply(lambda x: somatic_counts.get(x, 0))

##chang hotspots https://github.com/taylor-lab/hotspots/blob/master/publication_hotspots.vcf
hotspots = pd.read_csv(cwd / 'files' / 'hotspots.vcf', skiprows=1, sep='\t')
hotspots = hotspots[['#CHROM', 'POS', 'REF', 'ALT']].rename(columns={'#CHROM': 'Chromosome', 'POS': 'Start_Position', 'REF': 'Reference_Allele', 'ALT': 'Tumor_Seq_Allele2'})
hotspots.drop_duplicates(inplace=True)
hotspots['hotspot'] = True
merged = pd.merge(merged, hotspots, how='left', left_on=['CHROM', 'POS', 'REF_ALLELE', 'ALT_ALLELE'], right_on=['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])
unmerged = pd.merge(unmerged, hotspots, how='left', left_on=['CHROM', 'POS', 'REF_ALLELE', 'ALT_ALLELE'], right_on=['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])

##clinsig
annot['PRIVATE'] = (annot['INFO/dbsnp_CLNSIG_benign'] != True) | (annot['INFO/dbsnp_CLNORIGIN_somatic'] == True)
annot_dict = {i: j for i, j in zip(annot['ID'], annot['PRIVATE'].values)}
merged_privates = []
for i in merged.itertuples():
    temp = []
    id = i.ID.split(':')
    for position in range(len(i.REF_ALLELE)):
        temp.append(annot_dict.get(id[0] + ':' + str(int(id[1]) + position) + ':' + i.REF_ALLELE[position] + ':' + i.ALT_ALLELE[position], True))
    if True in temp:
        merged_privates.append(True)
    else:
        merged_privates.append(False)

merged['PRIVATE'] = merged_privates
unmerged['PRIVATE'] = unmerged['ID'].apply(lambda x: annot_dict.get(x, True))

print('clinsig')
print((len(unmerged.loc[(unmerged['LINEAGE'] == 'somatic') & unmerged['PRIVATE']]) +\
    sum(merged.loc[merged['LINEAGE'].isin(['somatic', 'both'])]['PRIVATE'])) / t_s)

print((t_g - (len(unmerged.loc[(unmerged['LINEAGE'] == 'germline') & unmerged['PRIVATE']]) +\
    sum(merged.loc[merged['LINEAGE'].isin(['germline', 'both'])]['PRIVATE']))) / t_g)

##exome metrics
annot['PRIVATE'] = (annot['INFO/gnomad_exomes_AF_popmax'] < .005) & (annot['INFO/gnomad_exomes_AC'] < 90)
annot_dict = {i: j for i, j in zip(annot['ID'], annot['PRIVATE'].values)}
merged_privates = []
for i in merged.itertuples():
    temp = []
    id = i.ID.split(':')
    for position in range(len(i.REF_ALLELE)):
        temp.append(annot_dict.get(id[0] + ':' + str(int(id[1]) + position) + ':' + i.REF_ALLELE[position] + ':' + i.ALT_ALLELE[position], True))
    if True in temp:
        merged_privates.append(True)
    else:
        merged_privates.append(False)

merged['PRIVATE'] = merged_privates
unmerged['PRIVATE'] = unmerged['ID'].apply(lambda x: annot_dict.get(x, True))

print('exome')
print((len(unmerged.loc[(unmerged['LINEAGE'] == 'somatic') & unmerged['PRIVATE']]) +\
    sum(merged.loc[merged['LINEAGE'].isin(['somatic', 'both'])]['PRIVATE'])) / t_s)

print((t_g - (len(unmerged.loc[(unmerged['LINEAGE'] == 'germline') & unmerged['PRIVATE']]) +\
    sum(merged.loc[merged['LINEAGE'].isin(['germline', 'both'])]['PRIVATE']))) / t_g)

##genome metrics
annot['PRIVATE'] = (annot['INFO/gnomad_genomes_AF_popmax'] < .009) & (annot['INFO/gnomad_genomes_AC'] < 26)
annot_dict = {i: j for i, j in zip(annot['ID'], annot['PRIVATE'].values)}
merged_privates = []
for i in merged.itertuples():
    temp = []
    id = i.ID.split(':')
    for position in range(len(i.REF_ALLELE)):
        temp.append(annot_dict.get(id[0] + ':' + str(int(id[1]) + position) + ':' + i.REF_ALLELE[position] + ':' + i.ALT_ALLELE[position], True))
    if True in temp:
        merged_privates.append(True)
    else:
        merged_privates.append(False)

merged['PRIVATE'] = merged_privates
unmerged['PRIVATE'] = unmerged['ID'].apply(lambda x: annot_dict.get(x, True))

print('genome')
print((len(unmerged.loc[(unmerged['LINEAGE'] == 'somatic') & unmerged['PRIVATE']]) +\
    sum(merged.loc[merged['LINEAGE'].isin(['somatic', 'both'])]['PRIVATE'])) / t_s)

print((t_g - (len(unmerged.loc[(unmerged['LINEAGE'] == 'germline') & unmerged['PRIVATE']]) +\
    sum(merged.loc[merged['LINEAGE'].isin(['germline', 'both'])]['PRIVATE']))) / t_g)

##tcga
print('tcga')
print((len(unmerged.loc[(((unmerged['somatic_count'] <= 87) | (unmerged['hotspot'] == True)) & (unmerged['LINEAGE'] == 'somatic'))]) +\
    len(merged.loc[(((merged['somatic_count'] <= 87) | (unmerged['hotspot'] == True)) & merged['LINEAGE'].isin(['somatic', 'both']))])) / t_s)

print((t_g - (len(unmerged.loc[((unmerged['germline_count'] <= 87) | (unmerged['hotspot'] == True)) & (unmerged['LINEAGE'] == 'germline')]) +\
              len(merged.loc[((merged['germline_count'] <= 87) | (merged['hotspot'] == True)) & merged['LINEAGE'].isin(['germline', 'both'])]))) / t_g)

##genome and exome and tcga
annot['PRIVATE'] = (annot['INFO/gnomad_genomes_AF_popmax'] < .009) & (annot['INFO/gnomad_genomes_AC'] < 26) & (annot['INFO/gnomad_exomes_AF_popmax'] < .005) & (annot['INFO/gnomad_exomes_AC'] < 90)
annot_dict = {i: j for i, j in zip(annot['ID'], annot['PRIVATE'].values)}
merged_privates = []
for i in merged.itertuples():
    temp = []
    id = i.ID.split(':')
    for position in range(len(i.REF_ALLELE)):
        temp.append(annot_dict.get(id[0] + ':' + str(int(id[1]) + position) + ':' + i.REF_ALLELE[position] + ':' + i.ALT_ALLELE[position], True))
    if True in temp:
        merged_privates.append(True)
    else:
        merged_privates.append(False)

merged['PRIVATE'] = merged_privates
unmerged['PRIVATE'] = unmerged['ID'].apply(lambda x: annot_dict.get(x, True))

print('complete')
print((len(unmerged.loc[(unmerged['LINEAGE'] == 'somatic') & ((unmerged['PRIVATE'] & (unmerged['somatic_count'] <= 87)) | (unmerged['hotspot'] == True))]) +\
    len(merged.loc[merged['LINEAGE'].isin(['somatic', 'both']) & ((merged['PRIVATE'] & (merged['somatic_count'] <= 87) | (merged['hotspot'] == True)))])) / t_s)

print((t_g - (len(unmerged.loc[(unmerged['LINEAGE'] == 'germline') & ((unmerged['PRIVATE'] & (unmerged['germline_count'] <= 87)) | (unmerged['hotspot'] == True))]) +\
    len(merged.loc[merged['LINEAGE'].isin(['germline', 'both']) & ((merged['PRIVATE'] & (merged['germline_count'] <= 87)) | (merged['hotspot'] == True))]))) / t_g)

unmerged['to_use'] = (unmerged['LINEAGE'] == 'somatic') & ((unmerged['PRIVATE'] & (unmerged['somatic_count'] <= 87)) | (unmerged['hotspot'] == True))
merged['to_use'] = merged['LINEAGE'].isin(['somatic', 'both']) & ((merged['PRIVATE'] & (merged['somatic_count'] <= 87) | (merged['hotspot'] == True)))
unmerged['to_use'] = unmerged['to_use'] | ((unmerged['LINEAGE'] == 'germline') & ((unmerged['PRIVATE'] & (unmerged['germline_count'] <= 87)) | (unmerged['hotspot'] == True)))
merged['to_use'] = merged['to_use'] | (merged['LINEAGE'].isin(['germline', 'both']) & ((merged['PRIVATE'] & (merged['germline_count'] <= 87)) | (merged['hotspot'] == True)))

final_df = pd.concat([unmerged.loc[unmerged['to_use'] == True], merged.loc[merged['to_use'] == True]], ignore_index=True)
final_df.drop(labels=['germline_count', 'somatic_count', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2', 'PRIVATE', 'to_use'], inplace=True, axis=1)

with open(file_path / 'germline' / 'data' / 'tumor_only_maf_filtered.pkl', 'wb') as f:
    pickle.dump(final_df, f)