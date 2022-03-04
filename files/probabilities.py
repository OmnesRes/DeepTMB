import pandas as pd
from tqdm import tqdm
import numpy as np
file_path = 'files/'
import pickle
from scipy.stats import binom
from sklearn.cluster import KMeans

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

clusters = 8
vaf_to_correct = .3
sample_df = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))

##https://gdc.cancer.gov/about-data/publications/pancanatlas
tcga_master_calls = pd.read_csv(cwd / 'files' / 'TCGA_mastercalls.abs_tables_JSedit.fixed.txt', sep='\t')
tcga_master_calls['sample_id'] = tcga_master_calls['sample'].apply(lambda x: x[:16])
tcga_master_calls = tcga_master_calls.loc[tcga_master_calls['call status'] == 'called']
tcga_master_calls = tcga_master_calls.groupby('sample_id')['purity'].mean().to_frame().reset_index()

sample_df = pd.merge(sample_df, tcga_master_calls, left_on='bcr_sample_barcode', right_on='sample_id', how='inner')

tcga_maf = pickle.load(open(cwd / 'files' / 'tcga_public_maf.pkl', 'rb'))
tcga_maf['vaf'] = tcga_maf['t_alt_count'].values / (tcga_maf['t_alt_count'].values + tcga_maf['t_ref_count'].values)
tcga_maf['depth'] = tcga_maf['t_ref_count'] + tcga_maf['t_alt_count']
tcga_maf['sample_id'] = tcga_maf['Tumor_Sample_Barcode'].apply(lambda x: x[:16])

cols = ['sample', 'subclonal.ix']
##https://gdc.cancer.gov/about-data/publications/pancan-aneuploidy
clonality_maf = pd.read_csv(cwd / 'files' / 'TCGA_consolidated.abs_mafs_truncated.fixed.txt', sep='\t', usecols=cols, low_memory=False)
result = clonality_maf.groupby('sample')['subclonal.ix'].apply(lambda x: sum(x) / len(x)).to_frame().reset_index()
del clonality_maf

sample_df = pd.merge(sample_df, result, left_on='Tumor_Sample_Barcode', right_on='sample', how='inner')

X = np.stack([sample_df.loc[sample_df['purity'] > .37]['purity'].values, sample_df.loc[sample_df['purity'] > .37]['subclonal.ix'].values], axis=-1)
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
mapping = {i: j for i, j in zip(sample_df.loc[sample_df['purity'] > .37]['Tumor_Sample_Barcode'].values, kmeans.labels_)}

sample_df['bin'] = sample_df['Tumor_Sample_Barcode'].apply(lambda x: mapping.get(x, -1))
tcga_maf = pd.merge(tcga_maf, sample_df[['Tumor_Sample_Barcode', 'bin']], on='Tumor_Sample_Barcode', how='inner')

def get_observed_vafs(read_depth, bin, density, vaf_bins):
    ##density is number of required vafs per bin, will search neighboring read depths to get more vafs
    vafs = tcga_maf.loc[(tcga_maf['bin'] == bin) & (tcga_maf['depth'] == read_depth)]['vaf'].values
    n = 1
    ##only care about density at lower vafs
    while sum(vafs < vaf_to_correct) / sum(vaf_bins < vaf_to_correct) < density:
        upper_vafs = tcga_maf.loc[(tcga_maf['bin'] == bin) & (tcga_maf['depth'] == read_depth + n)]['vaf'].values
        ##have to extrapolate vafs
        upper_vafs = np.round_(upper_vafs * (read_depth + n) * (read_depth / (read_depth + n)), 0) / read_depth
        vafs = np.concatenate([vafs, upper_vafs], axis=0)
        if read_depth - n >= 8:
            lower_vafs = tcga_maf.loc[(tcga_maf['bin'] == bin) & (tcga_maf['depth'] == read_depth - n)]['vaf'].values
            ##have to extrapolate vafs
            lower_vafs = np.round_(lower_vafs * (read_depth - n) * (read_depth / (read_depth - n)), 0) / read_depth
            vafs = np.concatenate([vafs, lower_vafs], axis=0)
        n += 1
        if n > 5:  ##limit max possible search
            break
    return vafs

def get_observed_noise(n, repeats, expected_data):
    noises = []
    for i in range(repeats):
        sampling = np.random.choice(np.array(list(expected_data.keys())), size=n, p=np.array(list(expected_data.values())) / sum(expected_data.values()))
        sampling_distribution = {i: 0 for i in expected_data}
        sampling_distribution.update({i: j / n for i, j in zip(*np.unique(sampling, return_counts=True))})
        assert list(sampling_distribution.keys()) == list(expected_data.keys())
        noises.append(np.abs(np.array(list(expected_data.values())) - np.array(list(sampling_distribution.values())))[:, np.newaxis] / np.array(list(expected_data.values()))[:, np.newaxis])
    return np.mean(noises, axis=0)

def filter(new_data, target_data, vaf_bins, min_data, noise):
    target_data = target_data - (target_data[:, np.newaxis] * noise)[:, 0]
    while True:
        temp_counts = np.zeros(shape=(len(new_data),))
        new_fractions = new_data / sum(new_data)
        old_sum = sum(new_data)
        for index in range(sum(vaf_bins < .3)):
            if new_fractions[index] < target_data[index]:
                temp_counts[index] = int(((sum(new_data) * target_data[index]) - new_data[index]) / (1 - target_data[index]))
            else:
                pass
        new_data = new_data + temp_counts
        if old_sum == sum(new_data):
            break
    return (sum(new_data) - min_data) / sum(new_data)

bins = list(range(clusters)) + [-1]
filters = [[]] * len(bins)
true_distributions = [[]] * len(bins)
for bin in bins:
    true = tcga_maf.loc[(tcga_maf['bin'] == bin) & (tcga_maf['depth'] > 250)]['vaf'].values
    ##round true distribution to nearest .01 and get counts
    true_distribution = {i: 0 for i in np.round_(np.arange(0, 1.01, .01), 2)}
    true_distribution.update({i: j for i, j in zip(*np.unique(np.round_(true, 2), return_counts=True))})
    ##first and last bins are half the width
    true_distribution[0.00], true_distribution[1.00] = true_distribution[0.00] * 2, true_distribution[1.00] * 2
    true_distributions[bin] = true_distribution

for read_depth in tqdm(range(8, 251)):
    x = np.arange(0, read_depth + 1)
    vaf_bins = np.unique(np.round(x / read_depth, 2))
    expected = [[]] * len(bins)
    observed = [[]] * len(bins)
    for bin in bins:
        distributions = np.array([binom.pmf(x, read_depth, p) for p in true_distributions[bin]])
        weighted = np.sum((np.array(list(true_distributions[bin].values())) / sum(true_distributions[bin].values()))[:, np.newaxis] * distributions, axis=0)
        ##the theoretical distribution is for expected alt counts, need to convert to expected rounded vafs
        expected_data = {}
        for index, i in enumerate(np.round_(x/read_depth, 2)):
            expected_data[i] = expected_data.get(i, []) + [weighted[index]]
        ##above a read depth of 100 some bins have more than one value
        expected_data = {i: np.sum(expected_data[i]) for i in expected_data}
        assert list(expected_data.keys()) == sorted(expected_data.keys())
        expected[bin] = expected_data
        vafs = get_observed_vafs(read_depth, bin, 30, vaf_bins)
        observed_data = {i: 0 for i in vaf_bins}
        observed_data.update({i: j for i, j in zip(*np.unique(np.round_(vafs, 2), return_counts=True))})
        assert list(observed_data.keys()) == list(vaf_bins)
        observed[bin] = observed_data
    ##need to find min amount of observed data
    min_data = min([sum(i.values()) for i in observed])
    for bin in bins:
        noise = get_observed_noise(min_data, 100, expected[bin])
        ##the observed data is counts while the expected is fractions
        new_data = np.array(list(observed[bin].values()))
        target_data = np.array(list(expected[bin].values()))
        filters[bin] = filters[bin] + [filter(new_data, target_data, vaf_bins, sum(observed[bin].values()), noise)]

import statsmodels.api as sm
lowess = sm.nonparametric.lowess

filter_dict = {}
max_depth = 251
for bin in bins:
    z = lowess(filters[bin], np.arange(8, max_depth), return_sorted=False, frac=.15)
    filter_dict[bin] = {i: j for i, j in zip(np.arange(8, max_depth), z)}

with open(cwd / 'files' / 'probabilities.pkl', 'wb') as f:
    pickle.dump([filter_dict, sample_df], f)
