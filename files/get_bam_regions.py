import pandas as pd
import pyranges as pr
import pickle
import numpy as np
import json
from liftover import get_lifter

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

##gdc data portal metadata files for TCGA WXS bams.  multiple files because only 10k can be added to the cart at a time.
with open(file_path / 'bams' / 'first_part.json', 'r') as f:
    metadata = json.load(f)

with open(file_path / 'bams' / 'second_part.json', 'r') as f:
    metadata += json.load(f)

with open(file_path / 'bams' / 'third_part.json', 'r') as f:
    metadata += json.load(f)

sample_to_file ={}
for i in metadata:
    sample_to_file[i['associated_entities'][0]['entity_submitter_id']] = i['file_id']

##need to choose random locations in the exome
gff = pd.read_csv('~/Desktop/ATGC2/files/Homo_sapiens.GRCh37.87.gff3',
                  sep='\t',
                  names=['chr', 'unknown', 'gene_part', 'start', 'end', 'unknown2', 'strand', 'unknown3', 'gene_info'],
                  usecols=['chr','gene_part', 'start', 'end', 'gene_info'],
                  low_memory=False)

chromosomes = list(map(lambda x: str(x), list(range(1, 23)) + ['X', 'Y']))
gff_cds_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'CDS') & gff['chr'].isin(chromosomes), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge()

##get assumed size of the most common kit: https://bitbucket.org/cghub/cghub-capture-kit-info/src/master/BI/vendor/Agilent/whole_exome_agilent_1.1_refseq_plus_3_boosters.targetIntervals.bed
agilent_df = pd.read_csv('~/Desktop/ATGC2/files/whole_exome_agilent_1.1_refseq_plus_3_boosters.targetIntervals.bed', sep='\t', low_memory=False, header=None)
kit_pr = pr.PyRanges(agilent_df.rename(columns={0: 'Chromosome', 1: 'Start', 2: 'End'})).merge()
kit_cds_pr = kit_pr.intersect(gff_cds_pr).merge()

intervals = kit_cds_pr.df.iloc[np.random.choice(np.arange(0, len(kit_cds_pr)), size=1000, replace=False), :]

mid_points = np.apply_along_axis(int, -1, (intervals['Start'].values + intervals['End'].values)[:, np.newaxis] / 2)
chrs = intervals['Chromosome'].values


converter = get_lifter('hg19', 'hg38')
regions = []
for chr, loc in zip(chrs, mid_points):
    position = converter[chr][loc]
    if position == []:
        pass
    else:
        position = str(position[0][1])
        regions.append('chr' + chr + ':' + position + '-' + position)


with open(file_path + '/bams/queries.pkl', 'wb') as f:
    pickle.dump([sample_to_file, regions], f)