# LSTM Gender Classifier
This is the first home work for advanced machine learning course, where we had to classify the gender based on the name of a person. In this experiment we fine tuned the lstm model to increase the accuracy of model, the accuracy of lstm on test data is increased from 80.21% to 81.52%.

Our dataset constists of names and the assigned genders, we first changed the labels from male, female, to 0,1, and changed the names to vector representations, for example, the name Elizabeth is converted to

[ 5 38 35 52 27 28 31 46 34  0  0  0  0  0  0]

We have made three models, one is basic lstm model and in basic lstm model we have 361 parameters, it can be seen as follows:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 5)           135       
_________________________________________________________________
lstm (LSTM)                  (None, 5)                 220       
_________________________________________________________________
dense (Dense)                (None, 1)                 6         
=================================================================
Total params: 361
Trainable params: 361
Non-trainable params: 0
_________________________________________________________________
The accuracy of lstm model is as follows:
163/163 [==============================] - 1s 2ms/step - loss: 0.4311 - accuracy: 0.8021
LSTM Baseline test loss, test acc: [0.43105417490005493, 0.8020843267440796]

Classical Neural Network is trained and it has given the following accuracy:
163/163 [==============================] - 0s 1ms/step - loss: 0.5106 - accuracy: 0.7572
Neural Network test loss, test acc: [0.5106410980224609, 0.7572279572486877]


Then we fine tuned the lstm, by increasing the hidden_state_size of lstm to 25, it has given us more accuracy then the our baseline model.
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_14 (Embedding)     (None, None, 5)           135       
_________________________________________________________________
lstm_12 (LSTM)               (None, 25)                3100      
_________________________________________________________________
dense_16 (Dense)             (None, 1)                 26        
=================================================================
Total params: 3,261
Trainable params: 3,261
Non-trainable params: 0
_________________________________________________________________

the accuracy of fine tuned model is as follows:
163/163 [==============================] - 1s 3ms/step - loss: 0.4076 - accuracy: 0.8153
LSTM Tuned test loss, test acc: [0.40758052468299866, 0.8152915239334106]

For more detailed information please have a look into the notebooks. Thanks
