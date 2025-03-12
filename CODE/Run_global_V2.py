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
MAX_SEGMENTS = 10  # ✅ On s'arrête après 6 fichiers (30 000 valeurs)
cdir = os.getcwd()
OUTPUT_FOLDER = os.path.join(cdir,"ECG_Data")
MODEL_FOLDER = os.path.join(cdir,"DataTreatment/Models")
TIMEOUT = 600 #temps max d'éxecution (5mn)
tr_data = np.load('/Users/octave/Desktop/Centrale/Centrale 3A/Projet3A/DataTreatment/database/clean_ecg.npy')
var = np.var(tr_data,axis=1)
ecart_type_data = np.mean(var**(1/2))

def center(sig):
    return (sig-np.mean(sig))/(np.var(sig)**(1/2))

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

def save_files():
    #sauvegardes en CSV
    Valeurs_file = os.path.join(OUTPUT_FOLDER, f"ECG_values_{MAX_SEGMENTS}.csv")
    ecg_df=pd.DataFrame(ecg_data)
    ecg_df.to_csv(Valeurs_file, index=False)
    print(f"Fichier {Valeurs_file} enregistré avec {len(ecg_df)} lignes.")

    time_file = os.path.join(OUTPUT_FOLDER, f"time_values_{MAX_SEGMENTS}.csv")
    time_df=pd.DataFrame(time_data)
    time_df.to_csv(time_file, index=False)
    print(f"Fichier {time_file} enregistré avec {len(time_df)} lignes.")

def show_transform(ecg):
    plt.plot(ecg,label='signal original')
    plt.plot(ecg - np.mean(ecg),label='signal centré (signal - moyenne(signal))')
    plt.title('signal centré')
    plt.legend()
    plt.show()
    
    ecg = center(ecg)
    plt.plot(ecg,label = 'signal centré réduit')
    plt.legend()
    plt.title('Normalisation du signal acquis: signal centré réduit') 
    plt.show()

    
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
                    segment_count = 0
                    save_files()
                    treat_results(ecg_data)
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
asyncio.run(scan_devices())
asyncio.run(main())





