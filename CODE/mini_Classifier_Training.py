import numpy as np
import pandas as pd
from os import name,getcwd
import os
import keras
from keras.layers import Dense, Input,Dropout, LeakyReLU, Conv1D, MaxPooling1D,BatchNormalization, Add, GlobalAveragePooling1D, Softmax
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint



#Training d'un classifier plus léger pour faire des tests

cdir = getcwd()
datadir = os.path.join(cdir, 'database')
# Charger les fichiers
datafile = np.load(os.path.join(datadir, 'clean_ecg.npy'))
df = pd.read_csv(os.path.join(datadir, 'data.csv'))

nb_ep = 10 #number of epochs
train_batch_percentage = 0.8 #pourcentage des datas qui est prise pr l'entrainement, le reste est en test

#créer le réseau de neurones
In = Input(shape=(datafile.shape[1],1))
class EntryBlock(keras.layers.Layer):
  def __init__(self):
    super(EntryBlock, self).__init__()
    self.Conv = Conv1D(32,5,padding='same',kernel_regularizer=l2(0.01),name='EB_Conv')
    self.BN = BatchNormalization(name = 'EB_BatchNormalization')
    self.LeakyReLU = LeakyReLU(name = 'EB_LeakyReLu')
    #self.MaxPool = MaxPooling1D(name = 'EB_MaxPool')

  def __call__(self, inputs):
    x = self.Conv(inputs)
    x = self.BN(x)
    x = self.LeakyReLU(x)
    #x = self.MaxPool(x)
    return x

class ResidualBlock(keras.layers.Layer):
  def __init__(self,nb):
    super(ResidualBlock, self).__init__()
    self.Conv1 = Conv1D(32,5,padding='same',kernel_regularizer=l2(0.01),name='RB_'+str(nb)+'_Conv1')
    self.BN1 = BatchNormalization(name = 'RB_'+str(nb)+'_BatchNormalization1')
    self.LeakyReLU1 = LeakyReLU(name = 'RB_'+str(nb)+'_LeakyReLu1')
    self.Dropout = Dropout(0.3,name = 'RB_'+str(nb)+'_Dropout')
    self.Conv2 = Conv1D(32,5,padding='same',kernel_regularizer=l2(0.01),name='RB_'+str(nb)+'_Conv2')

    self.LSTM = keras.layers.LSTM(32, return_sequences=True, name = 'RB_'+str(nb)+'_LSTM')
    #self.Dense = Dense(64,kernel_regularizer=l2(0.01),name = 'RB_'+str(nb)+'_Dense_after_LSTM')
    #self.LDropout = Dropout(0.2,name = 'RB_'+str(nb)+'_Light_Dropout')
    self.BN2 = BatchNormalization(name = 'RB_'+str(nb)+'_BatchNormalization2')
    self.LeakyRelu2 = LeakyReLU(name = 'RB_'+str(nb)+'_LeakyReLu')
    self.MaxPool = MaxPooling1D(2,name = 'RB_'+str(nb)+'_MaxPool')

  def __call__(self, inputs):
    x = self.Conv1(inputs)
    x = self.BN1(x)
    x = self.LeakyReLU1(x)
    x = self.Dropout(x)
    x = self.Conv2(x)
    x = x + inputs

    x = self.LSTM(x)
    #x = self.Dense(x)
    #x = self.LDropout(x)
    x = self.BN2(x)
    x = self.LeakyRelu2(x)
    x = self.MaxPool(x)
    return x

class ClassificationBlock(keras.layers.Layer):
  def __init__(self):
    super(ClassificationBlock, self).__init__()
    self.Conv0 = Conv1D(32,3,padding='same',kernel_regularizer=l2(0.01),name='CB_Conv0')
    self.BN0 = BatchNormalization(name = 'CB_BatchNormalization0')
    self.GAP = GlobalAveragePooling1D(name = 'CB_Flatten')
    self.D1 = Dense(64, kernel_regularizer=l2(0.01),name = 'CB_Dense1')
    self.LRL1 = LeakyReLU(name = 'CB_LeakyReLu1')
    self.D2 = Dense(32,kernel_regularizer=l2(0.01), name = 'CB_Dense2')
    self.LRL2 = LeakyReLU(name = 'CB_LeakyReLu2')
    self.D3 = Dense(3, kernel_regularizer=l2(0.01),name = 'CB_Dense3')
    self.LRL3 = LeakyReLU(name = 'CB_LeakyReLu3')
    self.softmax = Softmax(name = 'CB_Softmax')

  def __call__(self, inputs):
    x = self.Conv0(inputs)
    x = self.BN0(x)
    x = self.GAP(x)
    x = self.D1(x)
    x = self.LRL1(x)
    x = self.D2(x)
    x = self.LRL2(x)
    x = self.D3(x)
    x = self.LRL3(x)
    x = self.softmax(x)
    return x


Bl0 = EntryBlock()
Bl1 = ResidualBlock(1)
Bl2 = ResidualBlock(2)
Bl3 = ResidualBlock(3)
Out = Bl2(Bl1(Bl0(In)))
Out = ClassificationBlock()(Out)


lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.01,
    first_decay_steps=5,  # Restart toutes les 5 epochs
    t_mul=1.0,  # Cycles de même durée
    m_mul=0.8,  # LR max diminue légèrement après chaque cycle
    alpha=0.1)  # LR min = 10% du LR max

'''
lr_shcedule = keras.optimizers.schedules.ExponentialDecay(
  initial_learning_rate = 0.05,
  decay_steps = 3
  decay_rate = 0.5)
'''
opt = keras.optimizers.Adam(learning_rate = lr_schedule)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#possible que le model_checkpoint crée un arrêt dans l'apprentissage, l'accuracy ne progresse pas
'''
model_checkpoint = ModelCheckpoint(os.path.join(cdir,f'Classifier_{nb_ep}_epochs.keras'),
                                   monitor='val_loss', save_best_only=True, mode='max', verbose=1)
'''
M = keras.Model(inputs = In, outputs = Out)
M.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy','precision','recall'])

print('model created and compiled')
print(M.summary())

x = datafile[:]
y = df['ritmi'].to_numpy()
y = np.array([[int(x==i) for i in range(3)] for x in y])
D = list(zip(x,y))
np.random.shuffle(D)
x,y = zip(*D)
x = np.array(x)
y = np.array(y)

train_batch_size = int(train_batch_percentage*datafile.shape[0])
x_train = x[:train_batch_size]
y_train = y[:train_batch_size]

test_batch_size = datafile.shape[0] - train_batch_size
x_test = x[train_batch_size:]
y_test = y[train_batch_size:]

print(f'percentage of data used for training: {train_batch_percentage},\n'+
      f'x_test.shape = {x_test.shape}, y_test.shape = {y_test.shape},\n'+
      f'x_train.shape = {x_train.shape},y__train.shape = {y_train.shape}')

print('\nStarting training')
M.fit(x_train,y_train,batch_size=32,epochs=nb_ep,verbose=2,validation_data=(x_test,y_test))
M.save(os.path.join(cdir,f'mini_Classifier_{nb_ep}_epochs.keras'))
print('model saved under name:',cdir+f'/mini_Classifier_{nb_ep}_epochs.keras')
