from tensorflow import keras
from scripts.dataExtract import loadData

testPath = "data/test_eng.csv"
test_x, test_y = loadData(testPath)

lstm_baseline = keras.models.load_model("lstmBaseLineModel.h5")
results = lstm_baseline.evaluate(test_x, test_y, batch_size=128)
print("LSTM Baseline test loss, test acc:", results)

neuralNetwork = keras.models.load_model("classicalNeuralNet.h5")
results = neuralNetwork.evaluate(test_x, test_y, batch_size=128)
print("Neural Network test loss, test acc:", results)


lstm_tuned = keras.models.load_model("LSTMfineTuned.h5")
results = lstm_tuned.evaluate(test_x, test_y, batch_size=128)
print("LSTM Tuned test loss, test acc:", results)
