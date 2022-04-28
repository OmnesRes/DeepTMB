import pickle
import requests
from bs4 import BeautifulSoup

patients = []
ethnicities = []
cancers = ['LAML', 'ACC', 'BLCA', 'LGG', 'BRCA', 'CESC', 'CHOL', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'DLBC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THYM', 'THCA', 'UCS', 'UCEC', 'UVM']

for cancer in cancers:
    print(cancer)
    r = requests.get('http://52.25.87.215/TCGAA/cancertype.php?cancertype=' + cancer + '&pageSize=1500&page=1')
    soup = BeautifulSoup(r.content)
    rows = []
    for i in soup.find_all("tr"):
        rows.append(i)
    for i in rows:
        if i.find_all('td'):
            if i.find('td').text == 'patient':
                patients.append(i.find_all('td')[-1].text)
            for j in i.find_all('td'):
                if j.text in ['EA', 'AA', 'EAA', 'NA', 'OA']:
                    if len(i.find_all('td')) == 6:
                        ethnicities.append(i.find_all('td')[-2].text)

mapping = {i: j for i, j in zip(patients, ethnicities)}

with open('files/ethnicity.pkl', 'wb') as f:
    pickle.dump(mapping, f)