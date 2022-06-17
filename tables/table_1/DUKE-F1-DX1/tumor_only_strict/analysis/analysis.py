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
data = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'data' / 'data.pkl', 'rb'))
ancestry = pickle.load(open(cwd / 'files' / 'ethnicity.pkl', 'rb'))
germline_samples = pickle.load(open(cwd / 'files' / 'germline' / 'data' / 'germline_samples.pkl', 'rb'))

[data.pop(i) for i in list(data.keys()) if not data[i]]
[germline_samples.pop(i) for i in list(germline_samples.keys()) if germline_samples[i] < 400]
[data.pop(i) for i in list(data.keys()) if i[:12] not in germline_samples]

cutoff = np.percentile([i[-1] / (i[1] / 1e6) for i in data.values()], 98)
values = [i for i in data.values() if (i[-1] / (i[1] / 1e6)) < cutoff]
anc = np.array([ancestry.get(i[:12], 'OA') for i in data if (data[i][-1] / (data[i][1] / 1e6)) < cutoff])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])

X = np.array([i[-1] / (i[1] / 1e6) for i in values])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

print('counting')
print('NA')
print(round(np.sqrt(np.mean((Y - X)**2)), 2))
print(round(r2_score(Y, X), 2))
print(round(sum((Y >= 10) & (X >= 10)) / sum(Y >= 10), 2)) #recall
print(round(sum((Y >= 10) & (X >= 10)) / (sum((Y >= 10) & (X >= 10)) + sum((Y < 10) & (X >= 10))), 2)) #prec

print('linear prob')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'linear_prob_nonsyn_predictions.pkl', 'rb'))

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))



print('gmm prob')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_nonsyn_predictions.pkl', 'rb'))

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))


print('gmm hotspot')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_hotspots_predictions.pkl', 'rb'))

cutoff = np.percentile([(i[-1] - i[-2]) / (i[1] / 1e6) for i in data.values()], 98)
values = [i for i in data.values() if ((i[-1] - i[-2]) / (i[1] / 1e6)) < cutoff]
anc = np.array([ancestry.get(i[:12], 'OA') for i in data if ((data[i][-1] - data[i][-2]) / (data[i][1] / 1e6)) < cutoff])
anc_encoding = {'AA': 1, 'EA': 2, 'EAA': 3, 'NA': 4, 'OA': 0}
anc = np.array([anc_encoding[i] for i in anc])

X = np.array([(i[0] - i[-2]) / (i[1] / 1e6) for i in values])
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))


print('gmm syn')
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_syn_predictions.pkl', 'rb'))

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
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_syn_hotspots_predictions.pkl', 'rb'))

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
run_predictions, test_idx, values, losses = pickle.load(open(cwd / 'tables' / 'table_1' / 'DUKE-F1-DX1' / 'tumor_only_strict' / 'results' / 'gmm_syn_hotspots_ancestry_predictions.pkl', 'rb'))

cutoff = np.percentile([(i[0] - i[4]) / (i[1] / 1e6) for i in data.values()], 98)
values = [i for i in data.values() if ((i[0] - i[4]) / (i[1] / 1e6)) < cutoff]
Y = np.array([i[2] / (i[3] / 1e6) for i in values])

print(round(np.mean(losses), 2))
print(round(np.sqrt(np.mean((t.inv(np.concatenate(run_predictions)) - Y[np.concatenate(test_idx)])**2)), 2))
print(round(r2_score(Y[np.concatenate(test_idx)], t.inv(np.concatenate(run_predictions))), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) / sum(Y[np.concatenate(test_idx)] >= 10), 2))
print(round(sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10))
            / (sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] >= 10)) + sum((t.inv(np.concatenate(run_predictions)) >= 10) & (Y[np.concatenate(test_idx)] < 10))), 2))






