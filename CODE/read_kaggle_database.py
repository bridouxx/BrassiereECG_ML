import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import os
import pickle

#"arjunascagnetto/ptbxl-atrial-fibrillation-detection" pas complets
kaggle_url = "arjunascagnetto/ptbxl-atrial-fibrillation-detection"
#path = '/Users/octave/.cache/kagglehub/datasets/protobioengineering/mit-bih-arrhythmia-database-modern-2023/versions/2'
#datasave = path+"/datasave"

try:
    ECGs = pickle.load(open(datasave,'rb'))
    print('ECGs loaded')
except:
    path = kagglehub.dataset_download(kaggle_url)
    datasave = path+"/datasave"
    print("Path to dataset files: open", path)
    ECGs = []
    for name in os.listdir(path):

        df = pd.read_csv(path+'/'+name)
        ECGs.append(df)
    pickle.dump(ECGs,open(datasave,'wb'))

df = ECGs[0]
t = 0
df['time'] = 0.0
for i in range(1,len(df)):
    df.loc[i,'time'] = df.loc[i-1,'time'] + df.loc[i,'time_ms']
df.set_index('time',drop=True,inplace=True)
plt.plot(df['MLII'],label='MLII')
plt.plot(df['V1'],label='V1')
plt.legend()
plt.show()


import pickle
import os
ECGs = []
for name in os.listdir(path):
        df = pd.read_csv(path+'/'+name)
        ECGs.append(df)
    pickle.dump(ECGs,open(datasave,'wb'))
    
