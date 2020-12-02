import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding,Activation

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate


class probAI():

    #using https://towardsdatascience.com/how-to-convert-pandas-dataframe-to-keras-rnn-and-back-to-pandas-for-multivariate-regression-dcc34c991df9

    actTypes=8 #segun los datos hay 8 pero debería haber 6

    lookbackLSTM=12
    batchSize=3
    epochs=1

    def __init__(self):
        print("::::::::::::::::::::::")
        print("Probabilistic AI init")
        print("::::::::::::::::::::::")
        print("")

    def train(self,df,targetCol):

        y_col=targetCol

        test_size = int(len(df) * 0.1) # here I ask that the test data will be 10% (0.1) of the entire data
        train = df.iloc[:-test_size,:].copy() # the copy() here is important, it will prevent us from getting: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
        # Try using .loc[row_index,col_indexer] = value instead
        test = df.iloc[-test_size:,:].copy()


        X_train = train.drop(y_col,axis=1).copy()
        y_train = train[[y_col]].copy() # the double brakets here are to keep the y in dataframe format, otherwise it will be pandas Series

        Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
        Xscaler.fit(X_train)
        scaled_X_train = Xscaler.transform(X_train)
        Yscaler = MinMaxScaler(feature_range=(0, 1))
        Yscaler.fit(y_train)
        scaled_y_train = Yscaler.transform(y_train)
        scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)

        scaled_y_train = np.insert(scaled_y_train, 0, 0)
        scaled_y_train = np.delete(scaled_y_train, -1)

        n_input = 25 #how many samples/rows/timesteps to look in the past in order to forecast the next sample
        n_features= X_train.shape[1] # how many predictors/Xs/features we have to predict y
        b_size = 32 # Number of timeseries samples in each batch
        generator = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=n_input, batch_size=b_size)




        model=self.actModel(n_input,n_features)
        print("X_train",X_train)
        print(" y_train", y_train)
        generator = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=n_input, batch_size=b_size)

        print("generator",generator)

        model.fit_generator(generator,epochs=self.epochs)
        self.saveModel(model,"probAct")

        loss_per_epoch = model.history.history['loss']
        plt.plot(range(len(loss_per_epoch)),loss_per_epoch);
        #plt.show()

        #estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)
        #kfold = KFold(n_splits=10, shuffle=True)
        ##########################################################################################################################


        X_test = test.drop(y_col,axis=1).copy()
        scaled_X_test = Xscaler.transform(X_test)
        test_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(X_test)), length=n_input, batch_size=b_size)

        y_pred_scaled = model.predict(test_generator)
        y_pred = Yscaler.inverse_transform(y_pred_scaled)
        results = pd.DataFrame({'y_true':test[y_col].values[n_input:],'y_pred':y_pred.ravel().astype(int)})
        print(results)

        #plt.plot(results)
        plt.close('all')
        results.plot()
        plt.show()
        #train_size = int(len(dataset) * 0.67)
        #test_size = len(dataset) - train_size
        #y_categorical=keras.utils.to_categorical(dataset[targetCol].factorize())

    def actModel(self,n_input, n_features):

        #original model
        model = Sequential()
        model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(self.actTypes, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        """
        model = Sequential()
        model.add(LSTM(250, activation='relu', input_shape=(n_input, n_features)))
        #model.add(Dense(500, activation='relu'))
        #model.add(Dense(self.actTypes, activation='softmax'))
        model.add(Dense(self.actTypes, activation='relu'))
        #model.compile(optimizer='adam', loss='mse')
        #model.compile(loss='categorical_crossentropy', optimizer='adam')
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        model.summary()
        #print(model.summary())
        """

        """
        model = Sequential()
        #model.add(Embedding(n_features,n_input))
        #model.add(LSTM(64,activation='tanh'))#Create Input Layer
        model.add(Dense(8, input_dim=n_input, activation='relu'))
        model.add(Dense(self.actTypes, activation='softmax'))#Create output layer
        #model.add(Activation('softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        """

        """
        from numpy import zeros
        vocab_size=n_features
        embedding_matrix = zeros((n_features, 100))

        #deep_inputs = Input(shape=(n_input,))
        #embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
        #embedding_layer = Embedding(n_features,n_input)(deep_inputs)
        #flatten=Flatten()(embedding_layer)
        #LSTM_Layer_1 = LSTM(128,input_shape=n_input, n_features)(flatten)
        deep_inputs = Input(shape=(n_input, n_features))
        LSTM_Layer_1=LSTM(250, activation='relu', input_shape=(n_input, n_features))(deep_inputs)
        dense_layer_1 = Dense(self.actTypes, activation='sigmoid')(LSTM_Layer_1)
        model = Model(inputs=deep_inputs, outputs=dense_layer_1)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        """

        return model

    def saveModel(self,model,name):
        model.save('models/'+name)

if __name__ == "__main__":
    AI=probAI()

    data=pd.read_csv("data/PARSED/traindata.csv")

    #filter useful columns
    #dataset=data.drop(['name', 'duration',"placeid","dayofmonth","lat","lon"], axis = 1)
    dataset=data.drop(['name', 'duration',"placeid","dayofmonth"], axis = 1)
    print(dataset.dtypes)

    dataset["type"]=pd.Categorical(dataset['type'])
    dataset["type"]=dataset.type.cat.codes
    #pd.set_option('display.max_rows', dataset.shape[0]+1)
    print(dataset)
    #sys.exit()
    #plt.plot(dataset["dayofweek"])
    #plt.show()

    AI.train(dataset,"type")
