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

    actTypes=14 #segun los datos hay 8 pero debería haber 6

    lookbackLSTM=12
    batchSize=3
    epochs=500

    actY="type" #df col to predict

    def __init__(self):
        print("::::::::::::::::::::::")
        print("Probabilistic AI init")
        print("::::::::::::::::::::::")
        print("")

    def distanceTrain(self,df,targetCol):

        y_col=targetCol

        test_size = int(len(df) * 0.1) # here I ask that the test data will be 10% (0.1) of the entire data
        train = df.iloc[:-test_size,:].copy() # the copy() here is important, it will prevent us from getting: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
        # Try using .loc[row_index,col_indexer] = value instead
        test = df.iloc[-test_size:,:].copy()

        #X
        X_train = train.drop(y_col,axis=1).copy()
        y_train = train[[y_col]].copy() # the double brakets here are to keep the y in dataframe format, otherwise it will be pandas Series

        Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
        Xscaler.fit(X_train)
        scaled_X_train = Xscaler.transform(X_train)

        #Y
        Yscaler = MinMaxScaler(feature_range=(0, 1))
        Yscaler.fit(y_train)
        scaled_y_train = Yscaler.transform(y_train)
        scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)

        scaled_y_train = np.insert(scaled_y_train, 0, 0)
        scaled_y_train = np.delete(scaled_y_train, -1)

        n_input = 25 #how many samples/rows/timesteps to look in the past in order to forecast the next sample
        n_features= X_train.shape[1] # how many predictors/Xs/features we have to predict y
        b_size = 32 # Number of timeseries samples in each batch

        model=self.distanceModel(n_input,n_features)
        generator = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=n_input, batch_size=b_size)
        model.fit_generator(generator,epochs=10)

        self.saveModel(model,"probDistance")

        loss_per_epoch = model.history.history['loss']
        plt.plot(range(len(loss_per_epoch)),loss_per_epoch);
        plt.show()

    def actTrain(self,df):

        scaleY=False

        #:::::::::::::::::::::::::::::

        y_col=self.actY

        test_size = int(len(df) * 0.1) # here I ask that the test data will be 10% (0.1) of the entire data
        train = df.iloc[:-test_size,:].copy() # the copy() here is important, it will prevent us from getting: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
        # Try using .loc[row_index,col_indexer] = value instead
        test = df.iloc[-test_size:,:].copy()


        X_train = train.drop(y_col,axis=1).copy()
        y_train = train[[y_col]].copy() # the double brakets here are to keep the y in dataframe format, otherwise it will be pandas Series

        Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
        Xscaler.fit(X_train)
        scaled_X_train = Xscaler.transform(X_train)

        if scaleY:

            Yscaler = MinMaxScaler(feature_range=(0, 1))
            Yscaler.fit(y_train)
            scaled_y_train = Yscaler.transform(y_train)
            scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)

            scaled_y_train = np.insert(scaled_y_train, 0, 0)
            scaled_y_train = np.delete(scaled_y_train, -1)
        else:
            y_train=y_train.to_numpy()
            y_train=keras.utils.to_categorical(y_train, num_classes=self.actTypes)
            #print("y train",y_train)

        n_input = 25 #how many samples/rows/timesteps to look in the past in order to forecast the next sample
        n_features= X_train.shape[1] # how many predictors/Xs/features we have to predict y
        b_size = 32 # Number of timeseries samples in each batch

        model=self.actModel(n_input,n_features)
        print("scaled_X_train",scaled_X_train)
        print(" y_train", y_train)

        if scaleY:
            generator = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=n_input, batch_size=b_size)
        else:
            generator = TimeseriesGenerator(scaled_X_train, y_train, length=n_input, batch_size=b_size)


        if scaleY:
            model.fit_generator(generator,epochs=self.epochs)
        else:
            model.fit_generator(generator,epochs=self.epochs)
             #model.fit(scaled_X_train, y_train, batch_size=b_size)
        self.saveModel(model,"probAct")

        loss_per_epoch = model.history.history['loss']
        plt.plot(range(len(loss_per_epoch)),loss_per_epoch);
        plt.show()

        #estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)
        #kfold = KFold(n_splits=10, shuffle=True)
        ##########################################################################################################################


        X_test = test.drop(y_col,axis=1).copy()
        scaled_X_test = Xscaler.transform(X_test)
        test_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(X_test)), length=n_input, batch_size=b_size)

        if scaleY:
            y_pred_scaled = model.predict(test_generator)
            y_pred = Yscaler.inverse_transform(y_pred_scaled)
        else:
            y_pred = model.predict_classes(test_generator)

        print(y_pred)

        results = pd.DataFrame({'y_true':test[y_col].values[n_input:],'y_pred':y_pred.ravel().astype(int)})
        print(results)

        #plt.plot(results)
        plt.close('all')
        results.plot()
        plt.show()
        #train_size = int(len(dataset) * 0.67)
        #test_size = len(dataset) - train_size
        #y_categorical=keras.utils.to_categorical(dataset[targetCol].factorize())

    def distanceModel(self,n_input, n_features):
        model = Sequential()
        model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        return model

    def actModel(self,n_input, n_features):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(250, activation='relu', input_shape=(n_input, n_features), return_sequences=False))
        model.add(keras.layers.Dense(self.actTypes*2,  activation='relu'))
        model.add(keras.layers.Dense(self.actTypes, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.summary()

        return model

    def predictBlock(self,input):


        #predicts dayblock activity and location, needs the whole data to measure scaling correctly
        input=input.drop(['name', 'duration',"placeid","dayofmonth"], axis = 1)
        input["type"]=pd.Categorical(input['type'])
        input["type"]=input.type.cat.codes


        actModel=self.loadModel("probAct")

        y_col=self.actY

        #X
        X_input=input.drop(y_col,axis=1).copy()

        Xscaler = MinMaxScaler(feature_range=(0, 1))
        Xscaler.fit(X_input)
        #X_train = train.drop(y_col,axis=1).copy()
        #y_train = train[[y_col]].copy() # the double brakets here are to keep the y in dataframe format, otherwise it will be pandas Series

        Xpredict=X_input.tail(AI.lookbackLSTM*8)
        scaled_X_test = Xscaler.transform(Xpredict)

        p_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(Xpredict)), length=25, batch_size=32)

        y_pred= actModel.predict_classes(p_generator)



        print(y_pred)

        #predict activity


    def loadModel(self,name):
        return keras.models.load_model('models/'+name)

    def saveModel(self,model,name):
        model.save('models/'+name)

if __name__ == "__main__":
    AI=probAI()


    #activity train

    data=pd.read_csv("data/PARSED/acttypetraindata.csv")

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

    AI.actTrain(dataset)


    """
    #distance train
    data=pd.read_csv("data/PARSED/distancedtraindata.csv")
    print(data)
    #filter useful columns
    data=data.drop(['name'], axis = 1)
    print(data.dtypes)

    data["type"]=pd.Categorical(data['type'])
    data["type"]=data.type.cat.codes
    #pd.set_option('display.max_rows', dataset.shape[0]+1)
    print(data)
    #sys.exit()
    #plt.plot(dataset["dayofweek"])
    #plt.show()

    AI.distanceTrain(data,"distancefromlast")
    """

    #predict
    """
    data=pd.read_csv("data/PARSED/acttypetraindata.csv")
    print(data.tail(AI.lookbackLSTM))
    predicted=AI.predictBlock(data)
    print(predicted)
    """
