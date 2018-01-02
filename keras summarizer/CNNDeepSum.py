from keras.models import Sequential, model_from_json, save_model
from keras.layers import Dense, Activation, Embedding, Flatten, Convolution2D, pooling, Reshape, Dropout
from keras.engine import topology
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
import numpy as np

## Load Data and prepare input/output matrix

Title = np.load('TitleEnc.npy')
Top = np.load('TopEnc.npy')
Embeddings = np.array(np.load('Embeddings.npy'))
Lookup = np.load('Lookup.npy').item()
Inv = np.load('Inverselookup.npy').item()
TermFreq = np.load('TermFreq.npy').item()

TrainTitle = Title[0:82373,:]
TestTitle = Title[82373:83458,:]
ValidationTitle = Title[83458:84664,:]

TrainTop = Top[0:82373,:]
TestTop = Top[82373:83458,:]
ValidationTop = Top[83458:84664,:]

classW = []
for i in range(0,len(Inv)):
	classW.append(TermFreq[Inv[i]])

ClassW = {}
for i in range(0,len(Inv)):
		ClassW[i] = 1.0/classW[i]

ClassW[0] = 0.
ClassW[1] = 0.0000001
ClassW[2] = 0.
ClassW[3] = 1.0/len(Embeddings)


## Prepare Input
InputArray = []
PredictedArray = [] 
OutputArray = []

[TotalTrain,MaxLen] = TrainTitle.shape

for i in range(0,TotalTrain):
	for k in range(0,MaxLen-1):
		if(TrainTitle[i,k+1] != Lookup['<EOS>']):
			Pred = np.zeros(20, dtype = 'int')    
			InputArray.append(TrainTop[i])
			Pred[0:k+1] = TrainTitle[i,0:k+1]
			OutputArray.append(Pred)
			PredictedArray.append(TrainTitle[i,k+1])
		else:
			break 

TrainP = np.array(PredictedArray)
TrainO = np.array(OutputArray)
TrainI = np.array(InputArray)

## Prepare Test
InputArray = []
PredictedArray = [] 
OutputArray = []

[TotalTrain,MaxLen] = TestTitle.shape

for i in range(0,TotalTrain):
	for k in range(0,MaxLen-1):
		if(TestTitle[i,k+1] != Lookup['<EOS>']):
			Pred = Lookup['<PAD>']*np.ones(20, dtype = 'int')    
			InputArray.append(TestTop[i])
			Pred[0:k+1] = TestTitle[i,0:k+1]
			OutputArray.append(Pred)
			PredictedArray.append(TestTitle[i,k+1])
		else:			
			break 

TestP = np.array(PredictedArray)
TestO = np.array(OutputArray)
TestI = np.array(InputArray)

## Prepare Validation
InputArray = []
PredictedArray = [] 
OutputArray = []

[TotalTrain,MaxLen] = ValidationTitle.shape

for i in range(0,TotalTrain):
	for k in range(0,MaxLen-1):
		if(ValidationTitle[i,k+1] != Lookup['<EOS>']):
			Pred = np.zeros(20, dtype = 'int')    
			InputArray.append(ValidationTop[i])
			Pred[0:k+1] = ValidationTitle[i,0:k+1]
			OutputArray.append(Pred)
			PredictedArray.append(ValidationTitle[i,k+1])
		else:
			break 

ValidationP = np.array(PredictedArray)
ValidationO = np.array(OutputArray)
ValidationI = np.array(InputArray)



### Define the model
epo = 10
b_size = 100

InputModel = Sequential()
InputModel.add(Embedding(len(Embeddings), 300, weights=[Embeddings], input_length=80,trainable=False))
InputModel.add(Reshape((80,300,1), input_shape=(80,300)))
InputModel.add(Convolution2D(8, 5, 1, border_mode='valid',subsample=(2,1)))
InputModel.add(Dropout(0.2))
InputModel.add(Convolution2D(8, 7, 1, border_mode='valid', subsample=(3,1)))
InputModel.add(Dropout(0.2))
InputModel.add(Convolution2D(1,11, 1, border_mode='valid'))
InputModel.add(Dropout(0.2))
InputModel.add(Flatten())
#InputModel.add(Dense(30))
#InputModel.summary()

FeedbackModel = Sequential()
FeedbackModel.add(Embedding(len(Embeddings), 300, weights=[Embeddings], input_length=20,trainable=False))
FeedbackModel.add(Reshape((20,300,1), input_shape=(20,300)))
FeedbackModel.add(Convolution2D(8, 5, 1, border_mode='valid',subsample=(2,1)))
FeedbackModel.add(Dropout(0.2))
FeedbackModel.add(Convolution2D(8, 5, 1, border_mode='valid', subsample=(2,1)))
FeedbackModel.add(Dropout(0.2))
FeedbackModel.add(Convolution2D(1,2, 1, border_mode='valid'))
FeedbackModel.add(Dropout(0.2))
FeedbackModel.add(Flatten())
#FeedbackModel.add(Dense(30))
#FeedbackModel.summary()

OverallModel = Sequential()
OverallModel.add(topology.Merge([InputModel, FeedbackModel], mode='concat', concat_axis=1))
OverallModel.add(Dense(len(Embeddings), activation='softmax'))
OverallModel.summary()
sgd = SGD(lr=0.01, momentum=0., decay=0.0, nesterov=False)
OverallModel.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
from keras.utils.visualize_util import plot
plot(OverallModel, to_file='model.png')
plot(InputModel, to_file='model1.png')
plot(FeedbackModel, to_file='model2.png')
#OverallModel.load_weights('Models/11.hdf5')
CB = [ModelCheckpoint(filepath="./Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=2), EarlyStopping(monitor='val_loss', min_delta=0.005, patience=3, verbose=0, mode='auto')]
# we pass one data array per model input
#for i in range(0, epo):
#	OverallModel.fit([TrainI,TrainO], TrainP, nb_epoch=1, validation_data=([ValidationI, ValidationO], ValidationP), batch_size=b_size, callbacks=CB, class_weight=ClassW)
#save_model(OverallModel, '/home/parth/Documents/Codes/DeepSum/Models/model.h5')

