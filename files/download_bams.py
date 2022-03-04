import pickle
import requests
import pysam
import subprocess
import concurrent.futures
file_path = 'files/'

samples_to_file, regions = pickle.load(open(file_path + '/bams/queries.pkl', 'rb'))

##your token
with open(file_path + '/bams/gdc-user-token.2021-12-13T20_06_44.138Z.txt') as f:
    token = f.read()

def load_url(file, regions, name):
    payload = {
        "regions": regions
    }
    r = requests.post('https://api.gdc.cancer.gov/slicing/view/' + file,
                      json=payload,
                      headers={'X-Auth-Token': token,
                               "Content-Type": "application/json"})
    with open(file_path + 'bams/samples/' + name + '.bam', 'wb') as f:
        f.write(r.content)
    return len(r.content)

to_download = sorted(list(samples_to_file.keys()))


##some downloads fail
while True:
    cmd = ['ls', 'files/bams/samples/completed/']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    files = [str(i, 'utf-8').split('.bam')[0] for i in p.communicate()[0].split() if '.bam' in str(i)[-5:]]

    to_download = [i for i in to_download if i not in files]

    if to_download == []:
        break

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        future_to_url = {executor.submit(load_url, samples_to_file[sample], regions, sample): sample for sample in to_download}
        for future in concurrent.futures.as_completed(future_to_url):
            name = future_to_url[future]
            try:
                size = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (name, exc))
            else:
                print('%r was %d bytes' % (name, size))

    cmd = ['ls', file_path + '/bams/samples/']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    files = [str(i, 'utf-8') for i in p.communicate()[0].split() if '.bam' in str(i)[-5:]]

    for i in files:
        try:
            pysam.index(file_path + '/bams/samples/' + i)
            pysam.AlignmentFile(file_path + '/bams/samples/' + i, 'rb')
            subprocess.run('mv files/bams/samples/' + i + ' files/bams/samples/completed', shell=True)
            subprocess.run('mv files/bams/samples/' + i + '.bai files/bams/samples/completed', shell=True)
        except Exception as exc:
            print(exc)
        else:
            print(i)

