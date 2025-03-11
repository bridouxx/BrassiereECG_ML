import numpy as np
import asyncio
from bleak import BleakClient, BleakScanner
from bleak.exc import BleakDeviceNotFoundError
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import keras

# UUIDs BLE
SERVICE_UUID = "180D"
CHARACTERISTIC_UUID = "2A37"

# Configuration du stockage
DURATION = 5  # 10 secondes par fichier
SAMPLES_PER_SEGMENT = 5000  # 500 Hz * 10 secondes
MAX_SEGMENTS = 9  # ✅ On s'arrête après 6 fichiers (30 000 valeurs)
cdir = os.getcwd()
OUTPUT_FOLDER = os.path.join(cdir,"ECG_Data")
MODEL_FOLDER = os.path.join(cdir,"DataTreatment/Models")
TIMEOUT = 600 #temps max d'éxecution (5mn)
tr_data = np.load('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/DataTreatment/database/clean_ecg.npy')
acquired_signal_path = '/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/ECG_Data/ECG_values_10.csv'
acquired_time_path = '/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/ECG_Data/time_values_10.csv'
take_already_acquired_data = True

#Variance et ecart type training data
var = np.var(tr_data,axis=1)
ecart_type_data = np.mean(np.sqrt(var))
def rd_sig_ref():
    index_tr_ref = np.random.randint(0,len(tr_data)-1)
    sig_ref = tr_data[index_tr_ref]
    return sig_ref

#Make paths
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

try:
    Classifier = keras.saving.load_model(os.path.join(MODEL_FOLDER,'Classifier.keras'))
    VAE = keras.saving.load_model(os.path.join(MODEL_FOLDER,'VAE.keras'))
except:
    MODEL_FOLDER = input('could not find the models, please indicate the folder path')
    Classifier = keras.saving.load_model(os.path.join(MODEL_FOLDER,'Classifier.keras'))
    VAE = keras.saving.load_model(os.path.join(MODEL_FOLDER,'VAE.keras'))


print("Models Loaded")
print(VAE,Classifier)

ecg_data = np.zeros([MAX_SEGMENTS,SAMPLES_PER_SEGMENT])
time_data = np.zeros([MAX_SEGMENTS,SAMPLES_PER_SEGMENT])
segment_count = 0
sample_count = 0


def center(sig):
    return ecart_type_data*(sig-np.mean(sig))/(np.var(sig)**(1/2))

def save_files():
    #sauvegardes en CSV
    Valeurs_file = os.path.join(OUTPUT_FOLDER, f"ECG_values_{MAX_SEGMENTS}segments.csv")
    ecg_df=pd.DataFrame(ecg_data)
    ecg_df.to_csv(Valeurs_file, index=False)
    print(f"\nFichier {Valeurs_file} enregistré avec {len(ecg_df)} lignes.")

    time_file = os.path.join(OUTPUT_FOLDER, f"time_values_{MAX_SEGMENTS}segments.csv")
    time_df=pd.DataFrame(time_data)
    time_df.to_csv(time_file, index=False)
    print(f"Fichier {time_file} enregistré avec {len(time_df)} lignes.\n")

def save_treated_files(ecgs,time):
    Valeurs_file = os.path.join(OUTPUT_FOLDER, f"ECG_values_{MAX_SEGMENTS}_treated.csv")
    ecg_df=pd.DataFrame(ecgs)
    ecg_df.to_csv(Valeurs_file, index=False)
    print(f"\nFichier {Valeurs_file} enregistré avec {len(ecg_df)} lignes.")

    time_file = os.path.join(OUTPUT_FOLDER, f"time_values_{MAX_SEGMENTS}_treated.csv")
    time_df=pd.DataFrame(time)
    time_df.to_csv(time_file, index=False)
    print(f"Fichier {time_file} enregistré avec {len(time_df)} lignes.\n")

#show functions
def show_center(ecg):
    plt.plot(ecg,label='signal original')
    plt.plot(center(ecg),label='signal centré avec variance du training')
    plt.plot(rd_sig_ref(),label = "signal d'entrainement de référence")
    plt.title('Pré-traitement du signal acquis')
    plt.legend()
    plt.show()
    print(f'variance moyenne entrainement: {np.mean(var)}, variance signal centré: {np.var(center(ecg))}')

def show_interp(ecg,ecg_interp):
    plt.plot(ecg,label = 'signal orignial (centré réduit)')
    plt.plot(ecg_interp,label= 'signal interpolé pour les bon temps')
    plt.plot(rd_sig_ref(),label = "signal d'entrainement de référence")
    plt.title("Traitement d'interpolation")
    plt.legend()
    plt.show()

def show_VAE(ecg,ecg_VAE):
    plt.plot(ecg,label = 'signal orignial (interpolé)')
    plt.plot(ecg_VAE,label= 'signal traité par le VAE')
    plt.plot(rd_sig_ref(),label = "signal d'entrainement de référence")
    plt.title("Traitement par le VAE")
    plt.legend()
    plt.show()

def show_VAE2(ecg,ecg_VAE,time):
    plt.plot(time,ecg,label = 'signal orignial (interpolé)')
    plt.plot(time,ecg_VAE,label= 'signal traité par le VAE')
    plt.title("Traitement par le VAE")
    plt.legend()
    plt.show()
    
#treatment functions
def interp_ecg(ecg,time):
    time = time - time[0]
    new_time = np.linspace(0,time[-1],5000)
    interp_func = interp1d(time, ecg, kind='linear')
    ecg = interp_func(new_time)
    return ecg,new_time

def Treatment(ecg_data,time_data,show_steps=True):
    ##takes raw data acquired and do all steps of the treatment

    #pré-traitement (center)
    if show_steps:
        id_ex = np.random.randint(0,MAX_SEGMENTS-1)
        show_center(ecg_data[id_ex])
    ecgs = np.array([center(ecg) for ecg in ecg_data])

    #Interpolation
    ecgs_interp  = []
    new_time = []
    for i in range(len(ecgs)):
        ecg,time = interp_ecg(ecgs[i],time_data[i])
        ecgs_interp.append(ecg)
        new_time.append(time)
    ecgs_interp = np.array(ecgs_interp)
    new_time = np.array(new_time)
    
    if show_steps:
        show_interp(ecgs[id_ex],ecgs_interp[id_ex])
    ecgs = ecgs_interp[:]

    #traitement VAE
    print('treating signals with VAE...')
    ecgs_VAE = VAE(np.expand_dims(ecgs,2)).numpy()[:,:,0]
    
    if show_steps:
        show_VAE(ecgs[id_ex],ecgs_VAE[id_ex])
        show_VAE2(ecgs[id_ex],ecgs_VAE[id_ex],new_time[id_ex])
    ecgs = ecgs_VAE[:]
    
    #re-center
    ecgs = np.array([center(ecg) for ecg in ecgs])
    save_treated_files(ecgs,new_time)
    
    #traitement classifier
    print('treating signals with Classifier...')
    preds = Classifier(np.expand_dims(ecgs,2)).numpy()

    #moyennage diagnostique
    pred = np.sum(preds,axis=0)
    pred = pred/max(pred)
    
    #affichage diagnostique
    print(f'vector of prediction: {pred}')
    if max(pred)==pred[0]:
        result = "No arythmia"
    elif max(pred)==pred[2]:
        result = "Other form of arythmia"
    else:
        result = 'Atrial fibrilation'
    print(f'\nDiagnostic: {result}\n')
    
def treat_results(ecg_data):
    show_transform(ecg_data[0])
    ecg_data = [center(ecg_data[i]) for i in range(len(ecg_data))]
    ecg_data = np.expand_dims(ecg_data,2)
    cleaned_signals = VAE.predict(ecg_data)
    pred_clean = Classifier.predict(cleaned_signals)
    pred_sans_clean = Classifier.predict(ecg_data)

    pred_clean = np.sum(pred_clean,axis=0)/np.sum(pred_clean)
    pred_sans_clean = np.sum(pred_sans_clean,axis=0)/np.sum(pred_sans_clean)

    print(f'RESULT:\n prediction with VAE: {pred_clean}\n prediction without VAE:{pred_sans_clean}')
    if pred_clean[1]>0.4 or pred_clean[2]>0.4:
        if pred_clean[1]>pred_clean[2]:
            result = 'Atrial fibrilation'
        else:
            result = "Other form of arythmia"
    else:
        result = "No arythmia detected"

    print(f'This is my diagnostic: {result}')


async def notification_handler(sender, data):
    """Réception et traitement précis des données BLE"""
    decoded_data = data.decode("utf-8").strip()

    global ecg_data, time_data, segment_count, sample_count

    # Vérifie que la ligne contient une virgule (évite erreurs de format)
    if "," not in decoded_data:
        print(" Donnée ignorée (pas de virgule) : ", decoded_data)

    pairs = decoded_data.split(";")  # Séparer les valeurs reçues en paquets
    #print('len(pairs) =',len(pairs),f'pairs[0]={pairs[0]}')
    
    for pair in pairs:
        if "," in pair and pair.count(",") == 1:  # Vérifie que la donnée est bien au format
            time_val, ecg_value = pair.split(",")
    
            if ecg_value.strip():  # Vérifie que la valeur ECG n'est pas vide
                ecg_value = int(ecg_value)
                elapsed_time = float(time_val)
                    
                ecg_data[segment_count,sample_count] = ecg_value
                time_data[segment_count,sample_count] = time_val
                sample_count += 1
                
                if sample_count%500==0:
                    print('sample count',sample_count)

                if sample_count > SAMPLES_PER_SEGMENT-1:
                    print('captured segment',segment_count)
                    segment_count += 1
                    sample_count = 0

                if segment_count > MAX_SEGMENTS-1:
                    print(f"Capture terminée après {MAX_SEGMENTS} fichiers.")
                    save_files()
                    asyncio.get_running_loop().stop()


async def main():
    # Adresse MAC du Nano 33 BLE (remplace avec la bonne adresse)
    try:
        NANO_BLE_MAC = 'CB54C654-0C8C-D230-6698-DC31BCECCD81'
        async with BleakClient(NANO_BLE_MAC) as client:
            print(f"Connexion à {NANO_BLE_MAC}...")
            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
            await asyncio.sleep(TIMEOUT)  # Capture pendant 10 minutes, mais sera interrompu après 6 fichiers
            await client.stop_notify(CHARACTERISTIC_UUID)

    except BleakDeviceNotFoundError:
        NANO_BLE_MAC = input('enter device adress')
        async with BleakClient(NANO_BLE_MAC) as client:
            print(f"Connexion à {NANO_BLE_MAC}...")
            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
            await asyncio.sleep(600)  # Capture pendant 10 minutes, mais sera interrompu après 6 fichiers
            await client.stop_notify(CHARACTERISTIC_UUID)

async def scan_devices():
    devices = await BleakScanner.discover()
    for d in devices:
        print(f"Nom: {d.name}, Adresse: {d.address}")

#scan bluetooth and run
if not take_already_acquired_data:
    asyncio.run(scan_devices())
    asyncio.run(main())
else:
    ecg_data = pd.read_csv('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/ECG_Data/ECG_values_10.csv').to_numpy()
    time_data =  pd.read_csv('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/ECG_Data/time_values_10.csv').to_numpy()

Treatment(ecg_data,time_data)

    





