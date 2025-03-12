import numpy as np
import asyncio
from bleak import BleakClient, BleakScanner
import pandas as pd
import os
import time
import keras

# UUIDs BLE
SERVICE_UUID = "180D"
CHARACTERISTIC_UUID = "2A37"

# Adresse MAC du Nano 33 BLE (remplace avec la bonne adresse)
try:
    NANO_BLE_MAC = 'CB54C654-0C8C-D230-6698-DC31BCECCD81'
except:
    NANO_BLE_MAC = input('enter device adress')

# Configuration du stockage
DURATION = 10  # 10 secondes par fichier
SAMPLES_PER_SEGMENT = 5000  # 500 Hz * 10 secondes
MAX_SEGMENTS = 6  # ✅ On s'arrête après 6 fichiers (30 000 valeurs)
cdir = os.getcwd()
OUTPUT_FOLDER = os.path.join(cdir,"ECG_Data")
MODEL_FOLDER = os.path.join(cdir,"DataTreatment")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

if not os.path.exists(MODEL_FOLDER):
    try:
        MODEL_FOLDER = '/Users/alexandra/Documents/Models_folder'
        Classifier = keras.saving.load_model(os.path.join(MODEL_FOLDER,'Classifier.keras'))
        VAE = keras.saving.load_model(os.path.join(MODEL_FOLDER,'VAE.keras'))
    except:
        MODEL_FOLDER = input('could not find the models, please indicate the folder path')
        Classifier = keras.saving.load_model(os.path.join(MODEL_FOLDER,'Classifier.keras'))
        VAE = keras.saving.load_model(os.path.join(MODEL_FOLDER,'VAE.keras'))


print("Models Loaded")

global_data
segment_count = 0
start_time = time.time()
global_df = np.zeros([MAX_SEGMENTS,SAMPLES_PER_SEGMENT])
sample_count = 0  # ✅ Compteur d'échantillons pour chaque segment


async def notification_handler(sender, data):
    """Réception et traitement précis des données BLE"""
    decoded_data = data.decode("utf-8").strip()
    start_time = time.time()
    global_df = np.zeros([MAX_SEGMENTS,SAMPLES_PER_SEGMENT])
    sample_count = 0  # ✅ Compteur d'échantillons pour chaque segment

    # ✅ Vérifie que la ligne contient une virgule (évite erreurs de format)
    if "," not in decoded_data:
        print("⚠️ Donnée ignorée (pas de virgule) : ", decoded_data)

    pairs = decoded_data.split(";")  # Séparer les valeurs reçues en paquets
    print('len(pairs) =',len(pairs),pairs[0])
    
    for pair in pairs:
        if "," in pair and pair.count(",") == 1:  # Vérifie que la donnée est bien au format
            time_val, ecg_value = pair.split(",")
    
            if ecg_value.strip():  # Vérifie que la valeur ECG n'est pas vide
                #print(ecg_value)
                ecg_value = int(ecg_value)
                elapsed_time = float(time_val)

                new_data = pd.DataFrame([{"Time": elapsed_time, "ECG": ecg_value}])
                    
                global_df[segment_count,sample_count] = ecg_value
                sample_count += 1  # ✅ Incrémente le compteur d'échantillons
                if sample_count%100==0:
                    print('sample count',sample_count)

                    # ✅ Sauvegarde seulement quand on atteint 5000 valeurs
                if sample_count == SAMPLES_PER_SEGMENT-1:
                    print('captured segment',segment_count)
                    segment_count += 1
                    sample_count = 0

                if segment_count == MAX_SEGMENTS-1:
                        print(f"Capture terminée après {MAX_SEGMENTS} fichiers.")
                        Valeurs_file = os.path.join(OUTPUT_FOLDER, f"Valeurs.csv")
                        final_df=pd.DataFrame(global_df)
                        final_df.to_csv(Valeurs_file, index=False)
                        print(f"✅ Fichier {Valeurs_file} enregistré avec {len(final_df)} lignes.")
                        global_df = np.expand_dims(global_df,2)
                        cleaned_signals = VAE.predict(global_df)
                        pred_clean = Classifier.predict(cleaned_signals)
                        pred_sans_clean = Classifier.predict(global_df)

                        pred_clean = np.sum(pred_clean,axis=0)/np.sum(pred_clean)
                        pred_sans_clean = np.sum(pred_sans_clean,axis=0)/np.sum(pred_sans_clean)
                        
                        print(f'RESULT:\n prediction with VAE: {pred_clean}\n prediction without VAE:{pred_sans_clean}')
                        if pred_clean[1]>0.4 or pred_clean[2]>0.4:
                            if pred_clean[1]>pred_clean[2]:
                                result = 'Congratulations! You have Atrial Fibrilation!'
                            else:
                                result = "Hey! You've got AIDS!!"
                        else:
                            result = "Get out, you're boring me with your normal heart"

                        print(f'This is my diagnostic: {result}')
                        loop = asyncio.get_event_loop()
                        loop.stop()  # ✅ Arrête l'exécution d'asyncio proprement

async def main():

    async with BleakClient(NANO_BLE_MAC) as client:
        print(f"Connexion à {NANO_BLE_MAC}...")
        print('connected')
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        await asyncio.sleep(600)  # Capture pendant 10 minutes, mais sera interrompu après 6 fichiers
        await client.stop_notify(CHARACTERISTIC_UUID)

    #print(__name__)
    #loop = asyncio.get_event_loop()
    #print(loop)
    #loop.run_until_complete(main())

async def scan_devices():
    devices = await BleakScanner.discover()
    for d in devices:
        print(f"Nom: {d.name}, Adresse: {d.address}")

asyncio.run(scan_devices())

asyncio.run(main())
print(__name__)
