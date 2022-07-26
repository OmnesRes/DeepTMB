import pandas as pd
import pylab as plt
import seaborn as sns
import pickle
import pathlib

path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

file_path = cwd / 'files'
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]

loose_tumor_only_maf = pickle.load(open(file_path / 'germline' / 'data' / 'tumor_only_maf_filtered_loose.pkl', 'rb'))
loose_tumor_only_maf = loose_tumor_only_maf.loc[loose_tumor_only_maf['bcr_patient_barcode'].isin(germline_samples)]
moderate_tumor_only_maf = pickle.load(open(file_path / 'germline' / 'data' / 'tumor_only_maf_filtered_moderate.pkl', 'rb'))
moderate_tumor_only_maf = moderate_tumor_only_maf.loc[moderate_tumor_only_maf['bcr_patient_barcode'].isin(germline_samples)]
strict_tumor_only_maf = pickle.load(open(file_path / 'germline' / 'data' / 'tumor_only_maf_filtered_strict.pkl', 'rb'))
strict_tumor_only_maf = strict_tumor_only_maf.loc[strict_tumor_only_maf['bcr_patient_barcode'].isin(germline_samples)]

ancestry = pickle.load(open(file_path / 'ethnicity.pkl', 'rb'))

loose_germline_counts = loose_tumor_only_maf.loc[loose_tumor_only_maf['LINEAGE'] == 'germline']['bcr_patient_barcode'].value_counts().to_frame().reset_index()
loose_germline_counts['ancestry'] = loose_germline_counts['index'].apply(lambda x: ancestry.get(x, 'nan'))
loose_germline_counts = loose_germline_counts.loc[~(loose_germline_counts['ancestry'].isin(['nan', 'OA']))]
loose_germline_counts['filter'] = 'Permissive'
loose_copy = loose_germline_counts.copy()
loose_copy['ancestry'] = 'All'

strict_germline_counts = strict_tumor_only_maf.loc[strict_tumor_only_maf['LINEAGE'] == 'germline']['bcr_patient_barcode'].value_counts().to_frame().reset_index()
strict_germline_counts['ancestry'] = strict_germline_counts['index'].apply(lambda x: ancestry.get(x, 'nan'))
strict_germline_counts = strict_germline_counts.loc[~(strict_germline_counts['ancestry'].isin(['nan', 'OA']))]
strict_germline_counts['filter'] = 'Stringent'
strict_copy = strict_germline_counts.copy()
strict_copy['ancestry'] = 'All'

counts = pd.concat([loose_germline_counts, strict_germline_counts, loose_copy, strict_copy], ignore_index=True)
ancestry_counts = counts['ancestry'].value_counts().to_dict()
ancestry_counts = {i: int(ancestry_counts[i]/2) for i in ancestry_counts}


fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.9,
bottom=.19,
left=0.12,
right=1,
hspace=0.2,
wspace=0.2)
sns.violinplot(x='ancestry', y='bcr_patient_barcode', data=counts,
               split=True,
               hue='filter',
               ax=ax, order=['All', 'EA', 'NA', 'AA', 'EAA'], cut=0, inner=None, legend=False)
plt.setp(ax.collections, alpha=.5)
for i in range(0, 120, 20):
    ax.hlines(i, -.5, 4.5, linestyles='dotted', color='k', zorder=-1000, alpha=.3)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=12)
ax.set_xticklabels(['\nAll\n' + str(ancestry_counts['All']) + '\n104787',
                    'European\nAmerican\n' + str(ancestry_counts['EA']) + '\n64603',
                    'Native\nAmerican\n' + str(ancestry_counts['NA']) + '\n17720',
                    'African\nAmerican\n' + str(ancestry_counts['AA']) + '\n12487',
                    'East Asian\nAmerican\n' + str(ancestry_counts['EAA']) + '\n9977'], fontsize=12)
ax.tick_params(which='major', length=0, axis='x', pad=10)
ax.set_ylim(-.2, 100.2)
ax.set_xlabel('TCGA:\ngnomAD:     ', fontsize=12)
ax.xaxis.set_label_coords(0, -.15)
ax.set_ylabel('Private Variants', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_position(('outward', -15))
ax.spines['left'].set_bounds(0, 100)
plt.legend(frameon=False, loc=(.265, 1), ncol=2, title='Germline Filter Criteria', title_fontproperties={'size': 12})

plt.savefig(cwd / 'figures' / 'supplemental_figures' / 'figure_1' / 'results' / 'violin_plots.pdf')


