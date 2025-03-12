import numpy as np
import pandas as pd
from os import name,getcwd
import keras
from keras.layers import Dense, Input,Dropout, LeakyReLU, Conv1D, MaxPooling1D,BatchNormalization, Add, GlobalAveragePooling1D, Softmax


cdir = getcwd()
datadir = cdir+'/database'
datafile = np.load(datadir+'/clean_ecg.npy')
df = pd.read_csv(datadir+'/data.csv')

nb_ep = 12 #number of epochs
train_batch_percentage = 0.8 #pourcentage des datas qui est prise pr l'entrainement, le reste est en test

#créer le réseau de neurones
In = Input(shape=(datafile.shape[1],1))
class EntryBlock(keras.layers.Layer):
  def __init__(self):
    super(EntryBlock, self).__init__()
    self.Conv = Conv1D(64,5,padding='same',name='EB_Conv')
    self.BN = BatchNormalization(name = 'EB_BatchNormalization')
    self.LeakyReLU = LeakyReLU(name = 'EB_LeakyReLu')
    self.MaxPool = MaxPooling1D(name = 'EB_MaxPool')

  def __call__(self, inputs):
    x = self.Conv(inputs)
    x = self.BN(x)
    x = self.LeakyReLU(x)
    x = self.MaxPool(x)
    return x

class ResidualBlock(keras.layers.Layer):
  def __init__(self,nb):
    super(ResidualBlock, self).__init__()
    self.Conv1 = Conv1D(64,5,padding='same',name='RB_'+str(nb)+'_Conv1')
    self.BN1 = BatchNormalization(name = 'RB_'+str(nb)+'_BatchNormalization1')
    self.LeakyReLU1 = LeakyReLU(name = 'RB_'+str(nb)+'_LeakyReLu1')
    self.Dropout = Dropout(0.5,name = 'RB_'+str(nb)+'_Dropout')
    self.Conv2 = Conv1D(64,5,padding='same',name='RB_'+str(nb)+'_Conv2')

    self.LSTM = keras.layers.LSTM(64, return_sequences=True, name = 'RB_'+str(nb)+'_LSTM')
    self.BN2 = BatchNormalization(name = 'RB_'+str(nb)+'_BatchNormalization2')
    self.LeakyRelu2 = LeakyReLU(name = 'RB_'+str(nb)+'_LeakyReLu')
    self.MaxPool = MaxPooling1D(2,name = 'RB_'+str(nb)+'_MaxPool')

  def __call__(self, inputs):
    x = self.Conv1(inputs)
    x = self.BN1(x)
    x = self.LeakyReLU1(x)
    x = self.Dropout(x)
    x = self.Conv2(x)
    x = Add()([x,inputs])

    x = self.LSTM(x)
    x = self.BN2(x)
    x = self.LeakyRelu2(x)
    x = self.MaxPool(x)
    return x

class ClassificationBlock(keras.layers.Layer):
  def __init__(self):
    super(ClassificationBlock, self).__init__()
    self.Conv0 = Conv1D(32,3,padding='same',name='CB_Conv0')
    self.BN0 = BatchNormalization(name = 'CB_BatchNormalization0')
    self.GAP = GlobalAveragePooling1D(name = 'CB_Flatten')
    self.D1 = Dense(128, name = 'CB_Dense1')
    self.LRL1 = LeakyReLU(name = 'CB_LeakyReLu1')
    self.D2 = Dense(64, name = 'CB_Dense2')
    self.LRL2 = LeakyReLU(name = 'CB_LeakyReLu2')
    self.D3 = Dense(3, name = 'CB_Dense3')
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
Out = Bl3(Bl2(Bl1(Bl0(In))))
Out = ClassificationBlock()(Out)

M = keras.Model(inputs = In, outputs = Out)
M.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('model created and compiled')
print(M.summary())

x = datafile[:]
y = df['ritmi'].to_numpy()
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
M.save(cdir+f'/Classifier_{nb_ep}_epochs.keras')
print('model saved under name:',cdir+f'/Models/Classifier_{nb_ep}_epochs.keras')
