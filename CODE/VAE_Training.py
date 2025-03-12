import numpy as np
import pandas as pd
from os import name,getcwd
import keras
from keras.activations import sigmoid
from keras.layers import Dense, Input,Dropout, LeakyReLU, Conv1DTranspose, Conv1D, MaxPooling1D, LayerNormalization, Add, Flatten, Softmax

cdir = getcwd()
datadir = cdir+'/database'
clean_ecg = np.load(datadir+'/clean_ecg.npy')
noisy_ecg = np.load(datadir+'/noisy_ecg.npy')

nb_ep = 12 #number of epochs
train_batch_percentage = 0.8 #pourcentage des datas qui est prise pr l'entrainement, le reste est en test

#creating model
In = Input(shape=(clean_ecg.shape[1],1))
class EncoderBlock(keras.layers.Layer):
  def __init__(self,filters,name):
    super(EncoderBlock, self).__init__()
    self.Conv = Conv1D(filters = filters,kernel_size=16,
                       padding='same',name=name+'Conv')
    self.LN = LayerNormalization(name=name+'LN')
    self.ReLU = LeakyReLU(name=name+'ReLU')

  def __call__(self,x):
    x = self.Conv(x)
    x = self.LN(x)
    x = self.ReLU(x)
    return x

class DecoderBlock(keras.layers.Layer):
  def __init__(self,filters,name):
    super(DecoderBlock, self).__init__()
    self.DeConv = Conv1DTranspose(filters=filters,kernel_size=16,
                                  padding='same',name=name+'DeConv')
    self.LN = LayerNormalization(name=name+'LN')
    self.ReLU = LeakyReLU(name=name+'ReLU')

  def __call__(self,x):
    x = self.DeConv(x)
    x = self.LN(x)
    x = self.ReLU(x)
    return x

class AttentionBlock(keras.layers.Layer):
  def __init__(self,name):
    super(AttentionBlock, self).__init__()
    self.ConvAct = Conv1D(1,1,name=name+'ConvAct')
    self.ConvSaut = Conv1D(1,1,name=name+'ConvSaut')
    self.ReLU = LeakyReLU(name=name+'ReLU')
    self.Conv2 = Conv1D(1,1,name=name+'Conv2')

  def __call__(self,xatt,xsaut):
    xatt = self.ConvAct(xatt)
    xsaut = self.ConvSaut(xsaut)
    x = xatt+xsaut
    x = self.ReLU(x)
    x = self.Conv2(x)
    x = sigmoid(x)
    return x*xsaut

E0 = EncoderBlock(64,'E0')
E1 = EncoderBlock(64,'E1')
E2 = EncoderBlock(32,'E2')
E3 = EncoderBlock(32,'E3')

D0 = DecoderBlock(64,'D0')
D1 = DecoderBlock(64,'D1')
D2 = DecoderBlock(32,'D2')
D3 = DecoderBlock(32,'D3')

A0 = AttentionBlock('A0')
A1 = AttentionBlock('A1')
A2 = AttentionBlock('A2')
A3 = AttentionBlock('A3')

E = E3(E2(E1(E0(In))))
D = D3(E)
D = D + A3(D,E)
D = D2(D)
D = D + A2(D,E)
D = D1(D)
D = D + A1(D,E)
D = D0(D)
D = D + A0(D,E)

final_layer = Conv1DTranspose(filters=1,kernel_size=16,padding='same',
                              name=name+'DeConvFinal')
D = final_layer(D)

Model = keras.Model(inputs=In,outputs=D)
Model.compile(optimizer='adam',loss='mse',metrics=['MeanSquaredError','accuracy'])

print('model created and compiled')
Model.summary()

#Batching
x = np.expand_dims(noisy_ecg, axis=2)
y = np.expand_dims(clean_ecg, axis=2)
x = np.expand_dims(x, axis=3)
y = np.expand_dims(y, axis=3)
D = list(zip(x,y))
np.random.shuffle(D)
x,y = zip(*D)
x = np.array(x)
y = np.array(y)


train_batch_size = int(train_batch_percentage*noisy_ecg.shape[0])
x_train = x[:train_batch_size]
y_train = y[:train_batch_size]

test_batch_size = noisy_ecg.shape[0] - train_batch_size
x_test = x[train_batch_size:]
y_test = y[train_batch_size:]

print(f'percentage of data used for training: {train_batch_percentage},\n'+
      f'x_test.shape = {x_test.shape}, y_test.shape = {y_test.shape},\n'+
      f'x_train.shape = {x_train.shape},y__train.shape = {y_train.shape}')

print('\nStarting training')
#Model.fit(x_train,y_train,batch_size=32,epochs=10,verbose=2,validation_data=(x_test,y_test))

Model.save(cdir+f'/VAE_{nb_ep}_epochs.keras')
print('model saved under name:',cdir+f'/Models/VAE_{nb_ep}_epochs.keras')
