import numpy as np
import matplotlib.pyplot as plt
import keras
from scipy.interpolate import interp1d
import pandas as pd
import scipy

i=34
j = 4
facteur_div = 1.0

df = pd.read_csv('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/ECG_Data/ECG_values_10.csv')
df_time  = pd.read_csv('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/ECG_Data/time_values_10.csv')
tr_data = np.load('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/DataTreatment/database/clean_ecg.npy')
var = np.var(tr_data,axis=1)
ecart_type_data = np.mean(var**(1/2))

def center(sig):
    return np.sqrt(ecart_type_data)*(sig-np.mean(sig))/(np.var(sig)**(1/2))

ecgs = df.to_numpy()
ecgs = np.array([center(ecg) for ecg in ecgs])

size8 = int(len(ecgs[j])/facteur_div)
sig8 = ecgs[0,:size8]
x = np.linspace(0,size8-1,5000)
interp_func = interp1d(range(size8),sig8,kind='linear')
sig8 = interp_func(x)
plt.plot(sig8)
plt.plot(tr_data[i])
plt.show()

p_8 = scipy.signal.find_peaks(sig8, height=0.5, distance=100)[0]
p_tr = scipy.signal.find_peaks(tr_data[i], height=0.5, distance=100)[0]

print(len(p_8),len(p_tr),'facteur de division reccomandé:',len(p_8)/len(p_tr))
