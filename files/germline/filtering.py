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
maf = maf[['LINEAGE', 'CHROM', 'POS', 'End_Position', 'REF_ALLELE', 'ALT_ALLELE', 'ID', 'bcr_patient_barcode']]
germline_samples = maf.loc[maf['LINEAGE'].isin(['germline', 'both'])]['bcr_patient_barcode'].value_counts().to_dict()
with open(file_path / 'germline' / 'data' / 'germline_samples.pkl', 'wb') as f:
    pickle.dump(germline_samples, f)
t_s = sum(maf['LINEAGE'] == 'somatic') + sum(maf['LINEAGE'] == 'both')
t_g = sum(maf['LINEAGE'] == 'germline') + sum(maf['LINEAGE'] == 'both')

merged = maf.loc[(maf['REF_ALLELE'].str.len() > 1) & (maf['ALT_ALLELE'].str.len() > 1) & (maf['REF_ALLELE'].str.len() == maf['ALT_ALLELE'].str.len())]
unmerged = maf.loc[~((maf['REF_ALLELE'].str.len() > 1) & (maf['ALT_ALLELE'].str.len() > 1) & (maf['REF_ALLELE'].str.len() == maf['ALT_ALLELE'].str.len()))]

counts = unmerged['ID'].value_counts().to_dict()
unmerged['count'] = unmerged['ID'].apply(lambda x: counts.get(x, 0))

merged['merged_ID'] = merged['ID'] + '_' + merged['REF_ALLELE']
counts = merged['merged_ID'].value_counts().to_dict()
merged['count'] = merged['merged_ID'].apply(lambda x: counts.get(x, 0))

##chang hotspots https://github.com/taylor-lab/hotspots/blob/master/publication_hotspots.vcf
hotspots = pd.read_csv(cwd / 'files' / 'hotspots.vcf', skiprows=1, sep='\t')
hotspots = hotspots[['#CHROM', 'POS', 'REF', 'ALT']].rename(columns={'#CHROM': 'Chromosome', 'POS': 'Start_Position', 'REF': 'Reference_Allele', 'ALT': 'Tumor_Seq_Allele2'})
hotspots.drop_duplicates(inplace=True)
hotspots['hotspot'] = True
merged = pd.merge(merged, hotspots, how='left', left_on=['CHROM', 'POS', 'REF_ALLELE', 'ALT_ALLELE'], right_on=['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])
unmerged = pd.merge(unmerged, hotspots, how='left', left_on=['CHROM', 'POS', 'REF_ALLELE', 'ALT_ALLELE'], right_on=['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])

##clinsig
# annot['PRIVATE'] = (annot['INFO/dbsnp_CLNSIG_benign'] != True) | (annot['INFO/dbsnp_CLNORIGIN_somatic'] == True)
# annot_dict = {i: j for i, j in zip(annot['ID'], annot['PRIVATE'].values)}
# merged_privates = []
# for i in merged.itertuples():
#     temp = []
#     id = i.ID.split(':')
#     for position in range(len(i.REF_ALLELE)):
#         temp.append(annot_dict.get(id[0] + ':' + str(int(id[1]) + position) + ':' + i.REF_ALLELE[position] + ':' + i.ALT_ALLELE[position], True))
#     if True in temp:
#         merged_privates.append(True)
#     else:
#         merged_privates.append(False)
#
# merged['PRIVATE'] = merged_privates
# unmerged['PRIVATE'] = unmerged['ID'].apply(lambda x: annot_dict.get(x, True))
#
# print('clinsig')
# print((len(unmerged.loc[(unmerged['LINEAGE'] == 'somatic') & unmerged['PRIVATE']]) +\
#     sum(merged.loc[merged['LINEAGE'].isin(['somatic', 'both'])]['PRIVATE'])) / t_s)
#
# print((t_g - (len(unmerged.loc[(unmerged['LINEAGE'] == 'germline') & unmerged['PRIVATE']]) +\
#     sum(merged.loc[merged['LINEAGE'].isin(['germline', 'both'])]['PRIVATE']))) / t_g)

for filters in ['loose', 'moderate', 'strict']:
    if filters == 'loose':
        popmax = .01
    elif filters == 'moderate':
        popmax = .005
    else:
        popmax = .001
    ##exome metrics
    annot['PRIVATE'] = (annot['INFO/gnomad_exomes_AF_popmax'] < popmax) & (annot['INFO/gnomad_exomes_AF'] < popmax / 10)
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
    annot['PRIVATE'] = (annot['INFO/gnomad_genomes_AF_popmax'] < popmax) & (annot['INFO/gnomad_genomes_AF'] < popmax / 10)
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
    print((len(unmerged.loc[(((unmerged['count'] <= popmax * 10000) | (unmerged['hotspot'] == True)) & (unmerged['LINEAGE'] == 'somatic'))]) +\
        len(merged.loc[(((merged['count'] <= popmax * 10000) | (unmerged['hotspot'] == True)) & merged['LINEAGE'].isin(['somatic', 'both']))])) / t_s)

    print((t_g - (len(unmerged.loc[((unmerged['count'] <= popmax * 10000) | (unmerged['hotspot'] == True)) & (unmerged['LINEAGE'] == 'germline')]) +\
                  len(merged.loc[((merged['count'] <= popmax * 10000) | (merged['hotspot'] == True)) & merged['LINEAGE'].isin(['germline', 'both'])]))) / t_g)

    ##genome and exome and tcga
    annot['PRIVATE'] = (annot['INFO/gnomad_genomes_AF_popmax'] < popmax) & (annot['INFO/gnomad_genomes_AF'] < popmax / 10) & (annot['INFO/gnomad_exomes_AF_popmax'] < popmax) & (annot['INFO/gnomad_exomes_AF'] < popmax / 10)
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
    print((len(unmerged.loc[(unmerged['LINEAGE'] == 'somatic') & ((unmerged['PRIVATE'] & (unmerged['count'] <= popmax * 10000)) | (unmerged['hotspot'] == True))]) +\
        len(merged.loc[merged['LINEAGE'].isin(['somatic', 'both']) & ((merged['PRIVATE'] & (merged['count'] <= popmax * 10000) | (merged['hotspot'] == True)))])) / t_s)

    print((t_g - (len(unmerged.loc[(unmerged['LINEAGE'] == 'germline') & ((unmerged['PRIVATE'] & (unmerged['count'] <= popmax * 10000)) | (unmerged['hotspot'] == True))]) +\
        len(merged.loc[merged['LINEAGE'].isin(['germline', 'both']) & ((merged['PRIVATE'] & (merged['count'] <= popmax * 10000)) | (merged['hotspot'] == True))]))) / t_g)

    unmerged['to_use'] = (unmerged['PRIVATE'] & (unmerged['count'] <= popmax * 10000)) | (unmerged['hotspot'] == True)
    merged['to_use'] = (merged['PRIVATE'] & (merged['count'] <= popmax * 10000) | (merged['hotspot'] == True))

    final_df = pd.concat([unmerged.loc[unmerged['to_use'] == True], merged.loc[merged['to_use'] == True]], ignore_index=True)
    final_df.drop(labels=['count', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2', 'PRIVATE', 'to_use'], inplace=True, axis=1)

    with open(file_path / 'germline' / 'data' / ('tumor_only_maf_filtered_' + filters + '.pkl'), 'wb') as f:
        pickle.dump(final_df, f)

##metrics

##loose
# exome
# 0.9998369187771158
# 0.9555878655169971
# genome
# 0.9993898512867952
# 0.9473301560121047
# tcga
# 0.9999522003312236
# 0.9623812305523171
# complete
# 0.9992408287900216
# 0.98242445575252

##moderate
# exome
# 0.9995585560001237
# 0.9627936681278334
# genome
# 0.9973513360007423
# 0.9555436919199383
# tcga
# 0.9990299478983611
# 0.9728798358172182
# complete
# 0.9962378848927741
# 0.9862075236963611

##strict
# exome
# 0.9818445611006296
# 0.9759600299213456
# genome
# 0.9774301211018667
# 0.9696898857337937
# tcga
# 0.9931084124605302
# 0.9856709069484616
# complete
# 0.9651399827358843
# 0.9916778586809789


