# LSTM Gender Classifier
This is the first home work for advanced machine learning course, where we had to classify the gender based on the name of a person. In this experiment we fine tuned the lstm model to increase the accuracy of model, the accuracy of lstm on test data is increased from 80.21% to 81.52%.

Our dataset constists of names and the assigned genders, we first changed the labels from male, female, to 0,1, and changed the names to vector representations, for example, the name Elizabeth is converted to

[ 5 38 35 52 27 28 31 46 34  0  0  0  0  0  0]

Total params: 361
Trainable params: 361
Non-trainable params: 0

The accuracy of baseline lstm model is as follows:
LSTM Baseline test loss: 0.43105417490005493 , test acc: 0.8020843267440796

Classical Neural Network is trained and it has given the following accuracy:
Neural Network test loss: 0.5106410980224609, test acc: 0.7572279572486877


Then we fine tuned the lstm, by increasing the hidden_state_size of lstm to 25, it has given us more accuracy then the our baseline model, the accuracy of fine tuned model is as follows:
LSTM Tuned test loss: 0.40758052468299866, test acc: 0.8152915239334106

For more detailed information please have a look into the notebooks. Thanks

# Note: 
The structure for accessing data is the same mentioned in the home-work instructions, moreover, to build the model, run train.py in source, and then you will be able to test the model using test.py, but first you need to install the libraries mentioned in requirements.txt
