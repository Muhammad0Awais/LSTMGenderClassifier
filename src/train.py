import string

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Activation, Dropout, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from scripts.dataExtract import loadData


# %load_ext tensorboard 
# %tensorboard --logdir log   

trainPath = "data/train_eng.csv"

# Pre-processing data to find vector representations
train_x, train_y = loadData(trainPath)

# Creating vocabulary
unique = list(set("".join(string.ascii_lowercase[:26])))

unique.sort()
vocab = dict(zip(unique, range(1,len(unique)+1)))

# Splitting data into train and val
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y)

maxlen = 15
len_vocab = len(vocab)
# hyper-params

learningRate = 0.001
epoch = 100
hidden_state_size = 5

callback = EarlyStopping(monitor='val_loss', patience=15)
mc = ModelCheckpoint('lstm_baseline_model.h5', monitor='val_loss', mode='min', verbose=1)
reduce_lr_acc = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, verbose=1, min_delta=1e-4, mode='max')

def makemodel(maxlen, len_vocab, hidden_state_size, learningRate, lstm=True, fineTune = False):
  model = Sequential()
  if lstm:
    model.add(Embedding(input_dim=len_vocab+1, output_dim=5))
    model.add(LSTM(hidden_state_size, input_shape=(maxlen,len_vocab)))
    # if fineTune:
    #   model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
  else:
    model.add(Embedding(input_dim=len_vocab+1, output_dim=5, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(85, activity_regularizer=l2(0.002)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activity_regularizer=l2(0.002)))
    model.add(Activation('sigmoid'))
  
  model.compile(loss='BinaryCrossentropy', optimizer=Adam(learningRate),metrics=['accuracy'])
  return model


lstm_baseline_model = makemodel(maxlen=maxlen, len_vocab=len_vocab, hidden_state_size=hidden_state_size, learningRate=learningRate, lstm=True, fineTune=False)
lstm_baseline_model.summary()

batch_size = 64
tensorboard = TensorBoard(log_dir='log/{}'.format("LSTM Baseline Model")) 
history = lstm_baseline_model.fit(X_train, y_train, batch_size=batch_size, epochs=100, verbose=1, validation_data =(X_val, y_val), callbacks = [tensorboard])

lstm_baseline_model.save("lstmBaseLineModel.h5")

nn_baseline_model = makemodel(maxlen=maxlen, len_vocab=len_vocab, hidden_state_size=hidden_state_size, learningRate=learningRate, lstm=False)
nn_baseline_model.summary()

batch_size = 64
tensorboard = TensorBoard(log_dir='log/{}'.format("Neural Network Baseline Model")) 
history = nn_baseline_model.fit(X_train, y_train, batch_size=batch_size, epochs=100, verbose=1, validation_data =(X_val, y_val), callbacks=[callback, mc, reduce_lr_acc, tensorboard])

nn_baseline_model.save("classicalNeuralNet.h5")

# hyper-params

learningRate = 0.001
epoch = 100
batch_size = 64
hidden_state_size = 25

lstm_tuned_model = makemodel(maxlen=maxlen, len_vocab=len_vocab, hidden_state_size=hidden_state_size, learningRate=learningRate, lstm=True, fineTune=True)
lstm_tuned_model.summary()
batch_size = 64
tensorboard = TensorBoard(log_dir='log/{}'.format("LSTM Tuned Model")) 
history = lstm_tuned_model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1, validation_data =(X_val, y_val), callbacks=[callback, tensorboard]) #  mc, reduce_lr_acc,

lstm_tuned_model.save("LSTMfineTuned.h5")
