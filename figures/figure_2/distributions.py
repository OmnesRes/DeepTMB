import numpy as np
file_path = 'files/'
from tqdm import tqdm
import pickle
import concurrent.futures

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

depths = pickle.load(open(cwd / 'files' / 'depths.pkl', 'rb'))
probabilities, sample_df = pickle.load(open(cwd / 'files' / 'probabilities.pkl', 'rb'))
data = pickle.load(open(cwd / 'tables' / 'table_1' / 'MSK_468' / 'data' / 'data.pkl', 'rb'))

def get_distribution(sample):
    panel_distribution = []
    exome_distribution = []
    bin = sample_df.loc[sample_df['Tumor_Sample_Barcode'] == sample]['bin'].values[0]
    if sample not in data:
        return None
    if not data[sample]:
        return None
    for i in range(100):
        count = 0
        total = 0
        X = False
        while count < data[sample][2]:
            ##for zeros
            if count == data[sample][0] and not X:
                panel_distribution.append(total)
                X = True
            total += 1
            if np.random.random() > depths[sample][2]:
                depth = np.random.choice(depths[sample][0], size=1, p=depths[sample][1] / sum(depths[sample][1]))
                if depth >= 8:
                    if np.random.random() > probabilities[bin][int(depth)]:
                        count += 1
            else:
                count += 1
            if count == data[sample][0] and not X:
                panel_distribution.append(total)
                X = True
        exome_distribution.append(total)
    return [panel_distribution, exome_distribution]

distributions = {}
sample_df = sample_df[sample_df['Tumor_Sample_Barcode'].isin(depths)]
with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
    for sample, result in tqdm(zip(sample_df['Tumor_Sample_Barcode'].values, executor.map(get_distribution, sample_df['Tumor_Sample_Barcode'].values))):
        distributions[sample] = result

with open(cwd / 'figures' / 'figure_2' / 'distributions.pkl', 'wb') as f:
    pickle.dump(distributions, f)