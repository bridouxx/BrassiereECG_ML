import os
import pandas as pd
import numpy as np

cdir = os.getcwd()
datadir = cdir+'/database'

all_good = os.path.isfile(datadir+'/clean_ecg.npy')
all_good = all_good and os.path.isfile(datadir+'/noisy_ecg.npy')
all_good = all_good and os.path.isfile(datadir+'/data.csv')
if not all_good:
    print('missing some datafile, creating them')
    import data_pre_treatment.py

print('\n\n')

files = os.listdir(os.path.join(cdir,'Models')
# Vérifier si un fichier correspond au pattern "VAE_XX_epochs.keras"
matching_files = [f for f in files if f.startswith('VAE_') and f.endswith('_epochs.keras')]

if matching_files:
    print(f"File found: {matching_files[0]}, stop the program if this model is trained")
    time.sleep(6)

import VAE_Training.py

matching_files = [f for f in files if f.startswith('Classifier_') and f.endswith('_epochs.keras')]

if matching_files:
    print(f"File found: {matching_files[0]}, stop the program if this model is trained")
    time.sleep(6)
import Classifier_training.py
