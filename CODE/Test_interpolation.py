import numpy as np
import matplotlib.pyplot as plt
import keras
from scipy.interpolate import interp1d
import pandas as pd
import time as TIME

df = pd.read_csv('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/ECG_Data/ECG_values_10.csv')
df_time  = pd.read_csv('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/ECG_Data/time_values_10.csv')
tr_data = np.load('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/DataTreatment/database/clean_ecg.npy')
var = np.var(tr_data,axis=1)
ecart_type_data = np.mean(var**(1/2))

def center(sig):
    return (sig-np.mean(sig))/(np.var(sig)**(1/2))

def diagnose(pred_clean):
    #print(f'RESULT:\n prediction with VAE: {pred_clean}\n prediction without VAE:{pred_sans_clean}')
    pred_clean = pred_clean[0]
    if pred_clean[1]>0.4 or pred_clean[2]>0.4:
        if pred_clean[1]>pred_clean[2]:
            result = 'Atrial fibrilation'
        else:
            result = "Other form of arythmia"
    else:
        result = "No arythmia detected"

    print(f'This is my diagnostic: {result}')
    return result

for i in range(len(df)):
    ecg = df.iloc[i].to_numpy()
    ecg = center(ecg)
    time = df_time.iloc[i].to_numpy()
    time = time - time[0]
    diff = [time[i+1]-time[i] for i in range(len(time)-1)]
    new_time = np.linspace(0,time[-1],5000)
    interp_func = interp1d(time, ecg, kind='linear')
    ecg2 = interp_func(new_time)
    ecg2 = center(ecg2)

    def center(sig):
        return (ecart_type_data**(1/2))*(sig-np.mean(sig))/(np.var(sig)**(1/2))

    VAE = keras.saving.load_model('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/DataTreatment/Models/VAE.keras')
    C = keras.saving.load_model('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/DataTreatment/Models/Classifier.keras')


    print(f'treating signale {i} by VAE')
    ecg_clean = VAE(np.expand_dims(ecg,[0,2])).numpy()[0,:,0]
    ecg2_clean = VAE(np.expand_dims(ecg2,[0,2])).numpy()[0,:,0]

    ecg2_clean = center(ecg2_clean)
    ecg_clean = center(ecg_clean)
##    plt.plot(time,ecg_clean,label='après VAE sans interpolation')
##    plt.plot(new_time,ecg2_clean,label='après VAE avec interpolation')
##    #plt.plot(time,ecg,label='avant VAE')
##    plt.legend()
##    plt.title(f'Effet du VAE pour le signal {i}, avec et sans interpolation')
##    plt.ion()
##    plt.show()
##    TIME.sleep(2)
##    plt.close()
##    plt.ioff()

    print('treating by classifier')
    pred = C(np.expand_dims(ecg_clean,[0,2])).numpy()
    pred2 = C(np.expand_dims(ecg2_clean,[0,2])).numpy()

    res = diagnose(pred)
    res2 = diagnose(pred2)
    print(f'pred {i} without interpol:', pred,)
    print(f'{i} with interpol',pred2)
