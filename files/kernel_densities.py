import pickle
import numpy as np
import pysam
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import concurrent.futures

import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))


samples_to_file, regions = pickle.load(open(cwd / 'files' / 'bams' / 'queries.pkl', 'rb'))
files = sorted(list(samples_to_file.keys()))
files = [i for i in files if i.split('-')[3][:2] not in ['10', '11']]

def get_density(file):
    samfile = pysam.AlignmentFile(cwd / 'files' / 'bams' / 'samples' / 'completed' / (file + '.bam'), 'rb')
    counts = []
    for region in regions:
        count = 0
        for pileupcolumn in samfile.pileup(region.split(':')[0], int(region.split(':')[1].split('-')[0]) - 1, int(region.split(':')[1].split('-')[0]), truncate=True):
            for pileupread in pileupcolumn.pileups:
                if not pileupread.is_refskip:
                    count += 1
            counts.append(count)
    result = np.array(counts)
    result = result[result >= 8]
    if len(result) == 0:
        tail_prob = 0
        depths = 0
        probs = 0
    else:
        tail_prob = sum(result > 250) / len(result)
        result = result[result <= 250]
        kde = KernelDensity(bandwidth=4.64, kernel='gaussian')
        kde.fit(result[:, np.newaxis])
        x = np.linspace(0, max(result), 1000)
        logprob = kde.score_samples(x[:, np.newaxis])
        depths = np.arange(0, max(result) + 1)
        probs = np.exp(logprob[np.argmin(np.absolute(x - depths[:, np.newaxis]), axis=-1)])
    return [depths, probs, tail_prob, np.array(counts)]

depths = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    for file, result in tqdm(zip(files, executor.map(get_density, files))):
        depths[file] = result

with open(cwd / 'files' / 'depths.pkl', 'wb') as f:
    pickle.dump(depths, f)