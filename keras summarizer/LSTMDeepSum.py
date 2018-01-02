from keras.models import Sequential, model_from_json, save_model
from keras.layers import Dense, Activation, Embedding, Flatten, Convolution2D, pooling, Reshape, Dropout, TimeDistributed, Dense, RepeatVector, recurrent
from keras.engine import topology
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
import numpy as np
from keras.engine.training import slice_X
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, seed as random_seed, rand

random_seed(123) # Reproducibility

# Parameters for the model and dataset
NUMBER_OF_ITERATIONS = 20
epo = 10
RNN = recurrent.LSTM
INPUT_LAYERS = 2
OUTPUT_LAYERS = 2
AMOUNT_OF_DROPOUT = 0.3
b_size = 100
HIDDEN_SIZE = 300
INITIALIZATION = "he_normal" # : Gaussian initialization scaled by fan_in (He et al., 2014)
MAX_INPUT_LEN = 80
MIN_INPUT_LEN = 5
INVERTED = True
AMOUNT_OF_NOISE = 0.2
NUMBER_OF_CHARS = 100 # 75

## Load Data and prepare input/output matrix

Title = np.load('TitleEnc.npy')
Top = np.load('TopEnc.npy')
Embeddings = np.array(np.load('Embeddings.npy'))
Lookup = np.load('Lookup.npy').item()
Inv = np.load('Inverselookup.npy').item()
TermFreq = np.load('TermFreq.npy').item()

classW = []
for i in range(0,len(Inv)):
	classW.append(TermFreq[Inv[i]])

ClassW = {}
for i in range(0,len(Inv)):
	ClassW[i] = 1.0/classW[i]

TrainTitle = Title[0:82373,:]
TestTitle = Title[82373:83458,:]
ValidationTitle = Title[83458:84664,:]

TrainTop = Top[0:82373,:]
TestTop = Top[82373:83458,:]
ValidationTop = Top[83458:84664,:]

TrainO = np.array(TrainTitle)
SampleWeights = (TrainO > 0)*1.0
TrainO = np.expand_dims(TrainO, axis=2)
TrainI = np.array(TrainTop)


TestO = np.array(TestTitle)
TestI = np.array(TestTop)

ValidationO = np.expand_dims(np.array(ValidationTitle),axis=2)
ValidationI = np.array(ValidationTop)

# Model

model = Sequential()
model.add(Embedding(len(Embeddings), 300, weights=[Embeddings],trainable=False))
for layer_number in range(INPUT_LAYERS):
	model.add(recurrent.LSTM(HIDDEN_SIZE, init=INITIALIZATION, return_sequences=layer_number + 1 < INPUT_LAYERS))
	model.add(Dropout(AMOUNT_OF_DROPOUT))

model.add(RepeatVector(20))

for _ in range(OUTPUT_LAYERS):
	model.add(recurrent.LSTM(HIDDEN_SIZE, return_sequences=True, init=INITIALIZATION))
	model.add(Dropout(AMOUNT_OF_DROPOUT))

model.add(TimeDistributed(Dense(len(Embeddings), init=INITIALIZATION)))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'], sample_weight_mode="temporal")


from keras.utils.visualize_util import plot
plot(model, to_file='model3.png')

CB = [ModelCheckpoint(filepath="./Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1), EarlyStopping(monitor='val_loss', min_delta=0.005, patience=3, verbose=0, mode='auto')]
# we pass one data array per model input
#model.fit(TrainI, TrainO, nb_epoch=epo, validation_data=(ValidationI, ValidationO), batch_size=b_size, callbacks=CB, sample_weight = SampleWeights, class_weight=ClassW) 


#save_model(OverallModel, '/home/parth/Documents/Codes/DeepSum/Models/model.h5')

