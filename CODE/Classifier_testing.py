import numpy as np
import pandas as pd
from os import name,getcwd
import os
import keras
from keras.layers import Dense, Input,Dropout, LeakyReLU, Conv1D, MaxPooling1D,BatchNormalization, Add, GlobalAveragePooling1D, Softmax
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

cdir = getcwd()
datadir = os.path.join(cdir, 'database')
# Charger les fichiers
datafile = np.load(os.path.join(datadir, 'clean_ecg.npy'))
df = pd.read_csv(os.path.join(datadir, 'data.csv'))

nb_ep = 12 #number of epochs
train_batch_percentage = 0.8 #pourcentage des datas qui est prise pr l'entrainement, le reste est en test

Model = keras.saving.load_model(os.path.join(cdir,f'Classifier_{nb_ep}_epochs.keras'), custom_objects=None, compile=False)

lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.02,
    first_decay_steps=4,  # Restart toutes les 5 epochs
    t_mul=2,  # Cycles de même durée
    m_mul=0.6,  # LR max diminue légèrement après chaque cycle
    alpha=0.01)  # LR min = 10% du LR max

opt = keras.optimizers.Adam(learning_rate = lr_schedule)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model_checkpoint = ModelCheckpoint(os.path.join(cdir,f'Classifier_{nb_ep}_epochs.keras'),
                                   monitor='val_loss', save_best_only=True, mode='max', verbose=1)

Model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy','precision','recall'])

print('model loaded and compiled with specified metrics')
print(Model.summary())

x = datafile[:]
y = df['ritmi'].to_numpy()
y = np.array([[int(x==i) for i in range(3)] for x in y])

Model.evaluate(x=x,y=y,batch_size=32,verbose=2)
#Model.fit(x_train,y_train,batch_size=32,epochs=nb_ep,verbose=2,validation_data=(x_test,y_test),callbacks=[early_stopping, model_checkpoint])
Model.save(os.path.join(cdir,f'Classifier_{nb_ep}_epochs.keras'))
print('model saved under name:',cdir+f'/Classifier_{nb_ep}_epochs.keras')
