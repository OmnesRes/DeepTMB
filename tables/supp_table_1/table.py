import pickle
import pathlib
path = pathlib.Path.cwd()
if path.stem == 'DeepTMB':
    cwd = path
else:
    cwd = list(path.parents)[::-1][path.parts.index('DeepTMB')]
    import sys
    sys.path.append(str(cwd))

##tumor normal
gls = pickle.load(open(cwd / 'tables' / 'supp_table_1' / 'tumor_normal_gls_predictions.pkl', 'rb'))
gmm = pickle.load(open(cwd / 'tables' / 'supp_table_1' / 'tumor_normal_gmm_predictions.pkl', 'rb'))

##panel TMB
print(sum((gls['X'] > 10) & (gls['Y'] > 10)) / sum(gls['X'] > 10), sum(gls['X'] > 10))
print(sum((gls['X'] <= 10) & (gls['Y'] > 10)) / sum(gls['X'] <= 10), sum(gls['X'] <= 10))

##gls
print(sum((gls['median'] > 10) & (gls['Y'] > 10)) / sum(gls['median'] > 10), sum(gls['median'] > 10))
print(sum((gls['median'] <= 10) & (gls['Y'] > 10)) / sum(gls['median'] <= 10), sum(gls['median'] <= 10))

##gls 95%
print(sum((gls['lower'] > 10) & (gls['Y'] > 10)) / sum(gls['lower'] > 10), sum(gls['lower'] > 10))
print(sum((gls['upper'] < 10) & (gls['Y'] > 10)) / sum(gls['upper'] < 10), sum(gls['upper'] < 10))
print(sum(~((gls['lower'] > 10) | (gls['upper'] < 10)) & (gls['Y'] > 10)) / sum(~((gls['lower'] > 10) | (gls['upper'] < 10))), sum(~((gls['lower'] > 10) | (gls['upper'] < 10))))

##gmm
print(sum((gmm['median'] > 10) & (gmm['Y'] > 10)) / sum(gmm['median'] > 10), sum(gmm['median'] > 10))
print(sum((gmm['median'] <= 10) & (gmm['Y'] > 10)) / sum(gmm['median'] <= 10), sum(gmm['median'] <= 10))

##gmm
print(sum((gmm['lower'] > 10) & (gmm['Y'] > 10)) / sum(gmm['lower'] > 10), sum(gmm['lower'] > 10))
print(sum((gmm['upper'] < 10) & (gmm['Y'] > 10)) / sum(gmm['upper'] < 10), sum(gmm['upper'] < 10))
print(sum(~((gmm['lower'] > 10) | (gmm['upper'] < 10)) & (gmm['Y'] > 10)) / sum(~((gmm['lower'] > 10) | (gmm['upper'] < 10))), sum(~((gmm['lower'] > 10) | (gmm['upper'] < 10))))

##tumor only
gls = pickle.load(open(cwd / 'tables' / 'supp_table_1' / 'tumor_only_gls_predictions.pkl', 'rb'))
gmm = pickle.load(open(cwd / 'tables' / 'supp_table_1' / 'tumor_only_gmm_predictions.pkl', 'rb'))

##panel TMB
print(sum((gls['X'] > 10) & (gls['Y'] > 10)) / sum(gls['X'] > 10), sum(gls['X'] > 10))
print(sum((gls['X'] <= 10) & (gls['Y'] > 10)) / sum(gls['X'] <= 10), sum(gls['X'] <= 10))

##gls
print(sum((gls['median'] > 10) & (gls['Y'] > 10)) / sum(gls['median'] > 10), sum(gls['median'] > 10))
print(sum((gls['median'] <= 10) & (gls['Y'] > 10)) / sum(gls['median'] <= 10), sum(gls['median'] <= 10))

##gls 95%
print(sum((gls['lower'] > 10) & (gls['Y'] > 10)) / sum(gls['lower'] > 10), sum(gls['lower'] > 10))
print(sum((gls['upper'] < 10) & (gls['Y'] > 10)) / sum(gls['upper'] < 10), sum(gls['upper'] < 10))
print(sum(~((gls['lower'] > 10) | (gls['upper'] < 10)) & (gls['Y'] > 10)) / sum(~((gls['lower'] > 10) | (gls['upper'] < 10))), sum(~((gls['lower'] > 10) | (gls['upper'] < 10))))

##gmm
print(sum((gmm['median'] > 10) & (gmm['Y'] > 10)) / sum(gmm['median'] > 10), sum(gmm['median'] > 10))
print(sum((gmm['median'] <= 10) & (gmm['Y'] > 10)) / sum(gmm['median'] <= 10), sum(gmm['median'] <= 10))

##gmm
print(sum((gmm['lower'] > 10) & (gmm['Y'] > 10)) / sum(gmm['lower'] > 10), sum(gmm['lower'] > 10))
print(sum((gmm['upper'] < 10) & (gmm['Y'] > 10)) / sum(gmm['upper'] < 10), sum(gmm['upper'] < 10))
print(sum(~((gmm['lower'] > 10) | (gmm['upper'] < 10)) & (gmm['Y'] > 10)) / sum(~((gmm['lower'] > 10) | (gmm['upper'] < 10))), sum(~((gmm['lower'] > 10) | (gmm['upper'] < 10))))
