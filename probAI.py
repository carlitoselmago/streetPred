import pandas as pd
import pandas
import matplotlib.pyplot as plt

#keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class probAI():

    lookbackLSTM=6

    def __init__(self):
        print("::::::::::::::::::::::")
        print("Probabilistic AI init")
        print("::::::::::::::::::::::")
        print("")

    def train(self,dataset,targetCol):

        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size

        train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
        print("train,test sizes:",len(train), len(test))
        
        trainX=train.loc[:, train.columns != targetCol]
        trainY=train[targetCol]


        model=self.actModel(len(trainX.columns))
        print("trainX shape",trainX.shape)
        print("trainY shape",trainY.shape)
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    def actModel(self,Xsize):
        #model for predicting activity on a given date and timeblock
        model = Sequential()
        #model.add(LSTM(4, input_shape=(Xsize, self.lookbackLSTM)))
        model.add(LSTM(4,input_shape=(Xsize,)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        #print(model.summary())

        return model



if __name__ == "__main__":
    AI=probAI()

    data=pd.read_csv("data/PARSED/traindata.csv")

    #filter useful columns
    dataset=data.drop(['name', 'duration',"placeid","dayofmonth","lat","lon"], axis = 1)

    print(dataset.dtypes)

    dataset["type"]=pd.Categorical(dataset['type'])
    dataset["type"]=dataset.type.cat.codes
    print(dataset)
    #plt.plot(dataset["dayofweek"])
    #plt.show()

    AI.train(dataset,"type")
