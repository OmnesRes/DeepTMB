import pickle
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

sample_table = pickle.load(open(cwd / 'files' / 'tcga_public_sample_table.pkl', 'rb'))
data = pickle.load(open(cwd / 'figures' / 'figure_3' / 'data' / 'data.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))

[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if not data[i]]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]
nci_dict = {i: j for i, j in zip(sample_table['Tumor_Sample_Barcode'].values, sample_table['NCIt_tmb_label'].values) if j}
[data.pop(i) for i in list(data.keys()) if i not in nci_dict]

keys = [j for i,j in zip(data.values(), data.keys()) if (i[2] / (i[3] / 1e6)) <= 40]
nci = np.array([nci_dict[i] for i in data if (data[i][2] / (data[i][3] / 1e6)) <= 40])
class_counts = dict(zip(*np.unique(nci, return_counts=True)))
mask = [class_counts[i] >= 50 for i in nci]
keys = np.array(keys)[mask]
keys = [i[:12] for i in keys]

##all germline
np.mean([germline_samples[i] for i in keys])
##2451

loose_tumor_only_maf = pickle.load(open(file_path / 'germline' / 'data' / 'tumor_only_maf_filtered_loose.pkl', 'rb'))
loose_tumor_only_maf = loose_tumor_only_maf.loc[loose_tumor_only_maf['bcr_patient_barcode'].isin(keys)]
loose_tumor_only_maf = loose_tumor_only_maf.loc[loose_tumor_only_maf['LINEAGE'].isin(['germline', 'both'])]
loose_tumor_only_maf['bcr_patient_barcode'].value_counts().mean()
##43

moderate_tumor_only_maf = pickle.load(open(file_path / 'germline' / 'data' / 'tumor_only_maf_filtered_moderate.pkl', 'rb'))
moderate_tumor_only_maf = moderate_tumor_only_maf.loc[moderate_tumor_only_maf['bcr_patient_barcode'].isin(keys)]
moderate_tumor_only_maf = moderate_tumor_only_maf.loc[moderate_tumor_only_maf['LINEAGE'].isin(['germline', 'both'])]
moderate_tumor_only_maf['bcr_patient_barcode'].value_counts().mean()
##34

strict_tumor_only_maf = pickle.load(open(file_path / 'germline' / 'data' / 'tumor_only_maf_filtered_strict.pkl', 'rb'))
strict_tumor_only_maf = strict_tumor_only_maf.loc[strict_tumor_only_maf['bcr_patient_barcode'].isin(keys)]
strict_tumor_only_maf = strict_tumor_only_maf.loc[strict_tumor_only_maf['LINEAGE'].isin(['germline', 'both'])]
strict_tumor_only_maf['bcr_patient_barcode'].value_counts().mean()
##20