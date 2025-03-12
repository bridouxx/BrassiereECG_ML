import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.metrics import Precision, Recall, CategoricalAccuracy, F1Score
import os

#paths
def get_paths(base_path):
    if not os.path.isdir(base_path):
        base_path = input('folder not found, try again\n')

    p_models = os.path.join(base_path,'Models')
    p_database = os.path.join(base_path,'database')
    return p_models,p_database
    
def check_paths(p_models,p_database):
    return (os.path.isdir(p_models) and os.path.isdir(p_database))

base_path = os.getcwd()
p_models,p_database = get_paths(base_path)
if not check_paths(p_models,p_database):
    try:
        base_path = input('Enter path of the "DataTreatment" folder where there should be all data and models\n')
        p_models,p_database = get_paths(base_path)
        if not check_paths(p_models,p_database): raise ValueError('AAAAAAAAAARG!!')
        
    except:
        input('Could not find "Models" and/or "database" folder, create them, put the appropriate data in them and press "enter"\n')
        p_models,p_database = get_paths(base_path)

#load models and data
Classifier = keras.saving.load_model(os.path.join(p_models,'Classifier.keras'))
VAE = keras.saving.load_model(os.path.join(p_models,'VAE.keras'))

noisy_ecg = np.load(os.path.join(p_database,'noisy_ecg.npy'))
ecg = np.load(os.path.join(p_database,'clean_ecg.npy'))
df = pd.read_csv(os.path.join(p_database,'data.csv'))
labels = df['ritmi'].to_numpy()

#params
size_sample = 400
reset_pred = False

labels = labels[:size_sample]
noisy_ecg = np.expand_dims(noisy_ecg[:size_sample],2)
ecg = np.expand_dims(ecg[:size_sample],2)



#calculating models predictions
if reset_pred:
    print("Calculating model's output of the database")
    pred = Classifier.predict(ecg)
    npred = VAE.predict(noisy_ecg)
    npred = Classifier.predict(noisy_ecg)
    np.save(os.path.join(p_database,'prediction_clean.npy'),pred)
    np.save(os.path.join(p_database,'prediction_noisy.npy'),npred)
    
else:
    try:
        pred = np.load(os.path.join(p_database,'prediction_clean.npy'))
        npred = np.load(os.path.join(p_database,'prediction_noisy.npy'))

    except:
        print("Calculating model's output of the database") 
        pred = Classifier.predict(ecg)
        npred = VAE.predict(noisy_ecg)
        npred = Classifier.predict(noisy_ecg)
        np.save(os.path.join(p_database,'prediction_clean.npy'),pred)
        np.save(os.path.join(p_database,'prediction_noisy.npy'),npred)

#save originals
pred_original = pred[:]
npred_original = npred[:]

#calcul en un vect de vals qui valent 0, 1 ou 2
pred = [np.argmax(x) for x in pred]
npred = [np.argmax(x) for x in npred]

#calcul en vecteurs hot ones [0,0,1]
def heat(v):
    return [[int(x==i) for i in range(3)] for x in v]
y_true = heat(labels)
y_pred = heat(pred)
y_npred = heat(npred)

#calc accuracy matrix
def calc_acc(pred,lab):
    z = list(zip(pred,lab))
    acc = np.array([[np.sum([x==j and y==i for (x,y) in z]) for j in range(3)] for i in range(3)])
    acc = np.array([ac/sum(ac) for ac in acc])
    acc = np.trunc(acc*1000)/10
    return acc
    
def show_acc(acc,avec):
    # Création du DataFrame
    df = pd.DataFrame(acc,
                      index=['Sinus Rhythm', 'Atrial Fibrillation', 'Other Arrhythmias'],
                      columns=['Sinus Rhythm', 'Atrial Fibrillation', 'Other Arrhythmias'])
    df = df.astype(float)
    # Affichage du tableau
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(df, cmap='viridis')

    # Ajout des labels
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Sinus Rhythm', 'Atrial Fibrillation', 'Other Arrhythmias'], rotation=45)
    ax.set_yticklabels(['Sinus Rhythm', 'Atrial Fibrillation', 'Other Arrhythmias'])

    # Ajout des labels pour les dimensions
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Classe prédite', labelpad=20, fontsize=14, fontweight='bold')
    ax.set_ylabel('Classe réelle', labelpad=20, fontsize=14, fontweight='bold')
    ax.yaxis.set_label_position('left')

    # Ajout des valeurs dans les cellules
    for (i, j), val in np.ndenumerate(df):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    # Ajout d'une colorbar pour la légende
    fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.05)

    # Ajout d'un titre
    plt.title(f'Matrice de confusion {avec} bruit', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()

def show_metrics(y_pred,y_true):
    # Initialisation des métriques
    precision = Precision()
    recall = Recall()
    accuracy = CategoricalAccuracy()
    f1_score = F1Score()

    # Mise à jour des métriques avec les vraies classes et les classes prédites
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)
    accuracy.update_state(y_true, y_pred)
    f1_score.update_state(y_true, y_pred)

    # Calcul des résultats
    precision_result = precision.result().numpy()
    recall_result = recall.result().numpy()
    accuracy_result = accuracy.result().numpy()
    f1_score_result = f1_score.result().numpy()

    # Affichage des métriques
    print(f'Accuracy: {accuracy_result}')
    #print(f'Precision: {precision_result}')
    #print(f'Recall: {recall_result}')
    #print(f'F1 Score: {f1_score_result}')


print('\n\nClassification without noise----------------\n')
acc = calc_acc(pred,labels)
show_acc(acc,'sans')
show_metrics(y_pred,y_true)

print('\n\nClassification with noise----------------\n')
nacc = calc_acc(npred,labels)
show_acc(nacc,'avec')
show_metrics(y_npred,y_true)
