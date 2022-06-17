import numpy as np
import pickle
import pathlib
from model import utils
from sklearn.metrics import r2_score
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

t = utils.LogTransform(bias=4, min_x=0)

##counting metrics
data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'data' / 'data.pkl', 'rb'))
ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]

non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
non_syn_data = {i: sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) for i in data}
cutoff = np.percentile(list(non_syn_data.values()), 98)
values = [data[i] for i in data if non_syn_data[i] < cutoff]
non_syn_counts = [i for i in non_syn_data.values() if i < cutoff]
anc = np.array([ancestry.get(i[:12], 'OA') for i in non_syn_data if non_syn_data[i] < cutoff])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])

X = np.array([i / (j[1] / 1e6) for i, j in zip(non_syn_counts, values)])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

print('counting')
print('NA')
print(round(np.sqrt(np.mean((Y - X)**2)), 2))
print(round(r2_score(Y, X), 2))
print(round(sum((Y >= 10) & (X >= 10)) / sum(Y >= 10), 2)) #recall
print(round(sum((Y >= 10) & (X >= 10)) / (sum((Y >= 10) & (X >= 10)) + sum((Y < 10) & (X >= 10))), 2)) #prec


print('linear prob')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'linear_prob_nonsyn_predictions.pkl', 'rb'))

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))


print('gmm prob')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_nonsyn_predictions.pkl', 'rb'))

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))


print('gmm hotspot')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_hotspots_predictions.pkl', 'rb'))

non_syn_data = {i: sum([data[i][5].to_dict()[j] for j in data[i][5].index if j in non_syn]) - data[i][4] for i in data}
cutoff = np.percentile(list(non_syn_data.values()), 98)
values = [data[i] for i in data if non_syn_data[i] < cutoff]
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))


print('gmm syn')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_syn_predictions.pkl', 'rb'))

cutoff = np.percentile([i[0] / (i[1] / 1e6) for i in data.values()], 98)
values = [i for i in data.values() if (i[0] / (i[1] / 1e6)) < cutoff]
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))


print('gmm syn hotspots')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_syn_hotspots_predictions.pkl', 'rb'))

cutoff = np.percentile([(i[0] - i[4]) / (i[1] / 1e6) for i in data.values()], 98)
values = [i for i in data.values() if ((i[0] - i[4]) / (i[1] / 1e6)) < cutoff]
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))


print('gmm syn hotspots ancestry')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_normal' / 'results' / 'gmm_syn_hotspots_ancestry_predictions.pkl', 'rb'))

cutoff = np.percentile([(i[0] - i[4]) / (i[1] / 1e6) for i in data.values()], 98)
values = [i for i in data.values() if ((i[0] - i[4]) / (i[1] / 1e6)) < cutoff]
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))

