import pandas as pd
import numpy as np
import keras
import os.path
import math
import random
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from googledirections import *
from helpers import helpers
import json
from time import sleep
H=helpers()

#keras
import silence_tensorflow.auto
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
from sklearn import preprocessing


from distanceParser import distanceParser
from googleHistoryParser import _DAYSPLIT,catLocations, loadHistory

class probAI():
    #using https://towardsdatascience.com/how-to-convert-pandas-dataframe-to-keras-rnn-and-back-to-pandas-for-multivariate-regression-dcc34c991df9

    actTypes=15 #segun los datos hay 8 pero debería haber 6

    lookbackLSTM=12
    batchSize=3
    #epochs=500

    actY="type" #df col to predict

    def __init__(self):
        print("::::::::::::::::::::::")
        print("Probabilistic AI init")
        print("::::::::::::::::::::::")
        print("")


    def distanceTrain(self,df,epochs=10):
        print("train df")
        print(df)
        y_col="distancefromlast"

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

        print("scaled_y_train",scaled_y_train)

        model=self.distanceModel(n_input,n_features)
        generator = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=n_input, batch_size=b_size)
        model.fit_generator(generator,epochs=epochs)

        self.saveModel(model,"probDistance")

        loss_per_epoch = model.history.history['loss']
        plt.plot(range(len(loss_per_epoch)),loss_per_epoch);
        plt.show()

    def actTrain(self,df,epochsn=0):
        
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
        b_size = 10 # Number of timeseries samples in each batch

        model=self.actModel(n_input,n_features)

        if scaleY:
            generator = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=n_input, batch_size=b_size)
        else:
            generator = TimeseriesGenerator(scaled_X_train, y_train, length=n_input, batch_size=b_size)


        print("epochsn",epochsn)
        model.fit_generator(generator,epochs=epochsn)
    
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
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        return model

    def actModel(self,n_input, n_features):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(250, activation='relu', input_shape=(n_input, n_features), return_sequences=False))
        model.add(keras.layers.Dense(self.actTypes*2,  activation='relu'))
        model.add(Dropout(0.2))
        model.add(keras.layers.Dense(self.actTypes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.summary()

        return model

    def closestTypeLocation(self,lastloc,actType):
        closestDistance=1000000000000000000000.0
        closestLoc=False
        closestName=""
        for i,l in catLocations.iterrows():
            cloc=(l["latitudeE7"]/ 1e7,l["longitudeE7"]/1e7)
            if not math.isnan(cloc[0]):
                if actType==l["cat"]:
                    cdistance=self.distanceParser.calcDistance(lastloc,cloc)
                    if cdistance<closestDistance:
                        closestLoc=cloc
                        closestDistance=cdistance
                        closestName=l["name"]
        #print("closestLoc",closestLoc,"JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ")
        if closestLoc:
            return {"name":closestName,"lat":closestLoc[0],"lon":closestLoc[1]}
        else:
            return {"name":last["name"],"lat":float(last["lat"]),"lon":float(last["lon"])}


    def getPredLocation(self,lastloc,actType,distance,last):
        distance=abs(distance)
        #distance is in kilometers
        pad=0.1 #margin of distance

        lastloc=(float(lastloc[0]),float(lastloc[1]))

        if distance<=pad:
            #print("A",str(last["name"]))
            print("distancia es menor que pad-------")
            return {"name":(last["name"]),"lat":float(last["lat"]),"lon":float(last["lon"])}

        #define margins
        #m1=((loc[0]-distance),(loc[1]-distance))
        #m2=((loc[0]+distance),(loc[1]+distance))

        candidates=[]

        for i,l in catLocations.iterrows():
            cloc=(l["latitudeE7"]/ 1e7,l["longitudeE7"]/1e7)
            if not math.isnan(cloc[0]):

                cdistance=self.distanceParser.calcDistance(lastloc,cloc)
                if distance < (cdistance+pad) and distance > (cdistance-pad) and actType==l["cat"]:
                    print("added candidate",l["name"],"OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                    candidates.append({"name":l["name"],"lat":float(cloc[0]),"lon":float(cloc[1])})
                #if loc[0]<m1[0] and loc[1]>m1[1] and loc[0]<m2[0] and loc[1]<m2[1]
                if len(candidates)>4:
                    return random.choice(candidates)

        if len(candidates)>0:
            return candidates[-1]
        else:
            #return the closest location of type
            loc=self.closestTypeLocation(lastloc,actType)
            if loc:
                return {"name":loc["name"],"lat":float(loc["lat"]),"lon":float(loc["lon"])}
            else:
                return {"name":last["name"],"lat":float(last["lat"]),"lon":float(last["lon"])}
            #print(type(last["name"]))
            #print("B",str(last["name"].item()))

            #print(last)
            #return {"name":last["name"],"lat":float(last["lat"]),"lon":float(last["lon"])}

    def fillData(self,data,predicted):
        #fill data with trivial decisions
        lastRow=data.tail(1)

        lastDay=datetime(lastRow["year"],lastRow["month"],lastRow["dayofmonth"])
        nowDay=lastDay
        nowTimeBlock=int(lastRow["timeblock"])

        if nowTimeBlock>(_DAYSPLIT-1):
            #add 1 day
            nowDay+= timedelta(days=1)
            nowTimeBlock=0
        else:
            nowTimeBlock+=1

        predicted["name"]="?"
        predicted["dayofmonth"]=nowDay.day
        predicted["dayofweek"]=nowDay.weekday()
        predicted["month"]=nowDay.month
        predicted["year"]=nowDay.year
        predicted["timeblock"]=nowTimeBlock

        ###########################
        #CONFINAMIENTO HACK
        ###########################
        #if predicted["timeblock"]>4 or predicted["timeblock"]<2:
            #forzar confinamiento
        #    predicted["type"]="Home"

        locpred=self.getPredLocation((float(lastRow["lat"]),float(lastRow["lon"])),str(predicted["type"]),float(predicted["distancefromlast"]),lastRow)
        print("LOCPRED RESULT",locpred,"¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
        if isinstance(locpred["name"], pd.Series):
            name=locpred["name"].item()
        else:
            name=locpred["name"]
        #predicted["name"]="_"+str(name)
        predicted["name"]=str(name)

        predicted["lat"]=float(locpred["lat"])
        predicted["lon"]=float(locpred["lon"])

        print("predicting timeblock #",nowTimeBlock+1," of ",_DAYSPLIT, nowDay,"...")

        return predicted

    def timeblock2Hour(self,row):
        hour=int((row["timeblock"]*24)/_DAYSPLIT)-1
        if hour<0:
            hour=0
        fecha=datetime(row["year"],row["month"],row["dayofmonth"],int(hour))
        return fecha

    def getSimilarTransportType(self,originName,destName,distancefromlast,distanceToDestination):
        if originName==destName:
            print("SAME ORIGIN AND DESTINATION return false")
            return False
        #inputs: bike, bus, car, subway, plane, boat
        #output: driving, walking, bicycling, transit (metro etc)
        #candidates=self.historyData[self.historyData.apply(lambda row: {destination[0],destination[1]} == set((row.lat, row.lon)), axis=1)]
        #print(self.historyData)
        candidates=[]
        transport=""
        for i,l in self.historyData.iterrows():
            #print("destName",destName,'l["name"]',l["name"])
            #sleep(0.1)
            if destName==l["name"]:
                #print("LAST T MATCH")
                candidates.append(l["lasttransport"])


        if len(candidates)>0:
            #print("candidates",candidates)
            transport=H.most_frequent(candidates)

        #print("destination",destination)
        #print("GET SIMILAR TRANSPORT _------------------------------------")
        #print("candidates",candidates)
        #transport=candidates.lasttransport.mode().values[0]
        print("transport unproccesed",transport)
        print("distanceToDestination",distanceToDestination,"::::::::::::::::::::")
        if distanceToDestination>20.0:
            return "driving"

        if distanceToDestination>5.0:
            return "transit"

        if transport=="walking":
            return "walking"

        if transport=="bike":
            return "bicycling"

        if transport=="subway" or transport=="bus" or transport=="plane" or transport=="boat":
            return "transit"

        if transport=="car":
            return "driving"

       

        

        return "walking"

    def sample(self,preds, temperature=0.8):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def getDateFromstring(self,stringdate):
        parts=stringdate.split(",")
        try:
            return datetime(int(parts[0]),int(parts[1]),int(parts[2]))
        except:
            print("date start not detected, using last time block!")
            return False

    def predictBlocks(self,data,config):

        dateStart=self.getDateFromstring(config["date"])

        if dateStart:
            data_temp=data.copy()
            data_temp=data_temp.rename(columns={'dayofmonth': 'day'})
            data["date"]=  pd.to_datetime(data_temp[['year', 'month', 'day']],infer_datetime_format=True,errors='ignore')#pd.to_datetime((dataStr.year+dataStr.month+dataStr.dayofmonth),format='%Y%m%d')
            data["date"]= pd.to_datetime(data["date"].astype(str),format='%Y%m%d',errors='coerce')
            data = data[data.date < dateStart]
            data.drop(["date"], axis = 1)

        if bool(config["nohome"]):
            print("DELETING HOME!")
            data.drop(data.index[data['type'] == 'Home'], inplace=True)
            
        print(data)
        numblocks=int(config["days"])*_DAYSPLIT
        self.historyData=data
        self.distanceParser=distanceParser()
        data=self.distanceParser.parseData(data)
        predicted=[]
        routesTexts=[]
        points=[]
        routes=[]

        #hack para mostrar el punto de partida
        last=data.iloc[-1]
        points.append((last["lat"],last["lon"]))
        routesTexts.append('<h3><span class="numero">0</span>'+str(last["name"].replace("::::REST DAY::::","Hogar"))+"  "+str(self.timeblock2Hour(last))+'</h3>')
   
        for i in range(numblocks):
            pred=self.predictBlock(data,config)
            print("pred",pred)
            newRow=self.fillData(data,pred)
            last=data.iloc[-1]
            hour=int((last["timeblock"]*24)/_DAYSPLIT)-1
            if hour<0:
                hour=0

            lastDatetimeReal=datetime(last["year"],last["month"],last["dayofmonth"],int(hour))
            #lastDatetimeReal=dateStart
            lastDatetimeReal=lastDatetimeReal+timedelta(minutes=random.randint(0, 60))

            lastDatetime=datetime(2035,last["month"],last["dayofmonth"],int(hour))
            lastDatetime=lastDatetime+timedelta(minutes=random.randint(0, 60))

            route=False
            if abs(newRow["distancefromlast"])>0.001: #TODO: fix the abs trick
                distanceToDestination=self.distanceParser.calcDistance((last["lat"],last["lon"]),(newRow["lat"],newRow["lon"]) )
                transportMode=self.getSimilarTransportType(last["name"],newRow["name"],newRow["distancefromlast"],distanceToDestination)
                print("transportMode",transportMode)
                if transportMode:
                    route=groute(str(last["lat"])+","+str(last["lon"]),str(newRow["lat"])+","+str(newRow["lon"]),lastDatetime,transportMode)
                  
                    routesTexts.append("<h4>TRANSPORT: "+transportMode+"</h4>")

            else:
                route=False
           
            if route:

                newTime=lastDatetimeReal+timedelta(seconds=route[0]["legs"][0]["duration"]["value"])
            else:
                newTime=lastDatetimeReal

            if route:
                #place='<h3><span class="numero">'+str(len(points)+1)+'</span>'+str(newRow["name"])+"  "+str(newRow["dayofmonth"])+"."+str(newRow["month"])+"."+str(newRow["year"])+"  "+str(self.timeblock2Hour(newRow))+'</h3>'
                place='<h3><span class="numero">'+str(len(points)+1)+'</span>'+str(newRow["name"])+"  "+str(newTime.strftime("%m/%d/%Y, %H:%M:%S"))+'</h3>'
                if last["name"]!=newRow["name"]:

                    points.append((newRow["lat"],newRow["lon"]))
                    predicted.append(place)

                    if route:
                        routes.append(json.dumps(route[0]["legs"])+"\n")#+";"+"\n")
                        #predicted.append("TRANSITO ")#+str(newRow["distancefromlast"]))
                        for l in route[0]["legs"][0]["steps"]:
                            #}print(l["html_instructions"])
                            routesTexts.append(l["html_instructions"])
                            if "steps" in l:
                                for ii in l["steps"]:
                                    if "html_instructions" in ii:
                                        routesTexts.append(ii["html_instructions"])
                                    else:
                                        print("NO HTML INSTRUCTIONS FOUND!!")

                    routesTexts.append(place)
            data=data.append(newRow,ignore_index=True)
            #print("updated data")
           
            print(data.tail(i+1))


        center=H.centerOfLocs(points)
        H.buildPredictedHtml(center,routesTexts,points,routes,str(newTime.strftime("%m-%d-%Y, %H.%M.%S")))

        #return data.tail(blocks)
        return predicted


    def predictBlock(self,input,config):
        temperature=float(config["temperature"])
        lookBackLength=25

        originalInput=input.copy()

        #predicts dayblock activity and location, needs the whole data to measure scaling correctly, add target row as last with empty Y
        input=input.drop(['name',"year","lat","lon","distancefromlast","lasttransport"], axis = 1)

        #input["type"]=pd.Categorical(input['type'])
        #input["type"]=input.type.cat.codes
        print('input["type"]',input["type"])
        input["type"]=self.toCategorical(input["type"])
        #input["type"]=input["type"].astype('category')

        actModel=self.loadModel("probAct")

        y_col=self.actY

        #X
        X_input=input.drop(y_col,axis=1).copy()

        Xscaler = MinMaxScaler(feature_range=(0, 1))
        Xscaler.fit(X_input)
        #X_train = train.drop(y_col,axis=1).copy()
        #y_train = train[[y_col]].copy() # the double brakets here are to keep the y in dataframe format, otherwise it will be pandas Series

        #Xpredict=X_input.tail(AI.lookbackLSTM*4)
        Xpredict=X_input.tail(lookBackLength)
        scaled_X_test = Xscaler.transform(Xpredict)

        #p_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(Xpredict)), length=25, batch_size=32)
        p_generator = TimeseriesGenerator(scaled_X_test, np.zeros(lookBackLength), length=1, batch_size=32)
        #print("Xpredict",Xpredict)
        ActProb= actModel.predict(p_generator) #tokenized prediction

        #we get probabilities, sample with temperature
        ActPred=[]
        for a in ActProb:
            """
            print("::::::::::::::::::::::::::::::::::::::::::::")
            print(temperature)
            print(a)
            """
            chosen=self.sample(a,temperature) #XXXX
            #dist = RelaxedOneHotCategorical(temperature, logits=a)
            #print(chosen)
            #print("")
            ActPred.append(chosen)

        print("ActPred",ActPred)
        ActString=self.inverseCategorical(ActPred)


        #predict distance/location #############################################
        distanceModel=self.loadModel("probDistance")


        input=originalInput.copy()


        input=input.drop(['name','lat','lon','lasttransport'], axis = 1)

        input["type"]=self.toCategorical(input["type"])

        y_col="distancefromlast"

        X_train = input.drop(y_col,axis=1).copy()


        Xscaler.fit(X_train)

        y_train = input[[y_col]].copy()
        Yscaler = MinMaxScaler(feature_range=(0, 1))
        Yscaler.fit(y_train)


        Xpredict=X_train.tail(lookBackLength)

        scaled_X_test = Xscaler.transform(Xpredict)

        p_generator = TimeseriesGenerator(scaled_X_test, np.zeros(lookBackLength), length=1, batch_size=32)

        y_pred_scaled = distanceModel.predict(p_generator)

        distancePred = Yscaler.inverse_transform(y_pred_scaled)

        return {"type":ActString[-1],"distancefromlast":float(distancePred[-1][0])}

    def toCategorical(self,inputs):
        actle = preprocessing.LabelEncoder()
        actle.fit(inputs)
        self.actle=actle
        return actle.transform(inputs)

    def inverseCategorical(self,inputs):
        return self.actle.inverse_transform(inputs)
       

    def loadModel(self,name):
        return keras.models.load_model('models/'+name)

    def saveModel(self,model,name):
        model.save('models/'+name)

if __name__ == "__main__":
    AI=probAI()

    #activity train

    if not os.path.isdir("models/probAct"):
        data=pd.read_csv("data/PARSED/acttypetraindata.csv")

        #filter useful columns
        dataset=data.drop(['name', 'duration',"placeid","lat","lon"], axis = 1)
        #dataset=data.drop(['name', 'duration',"placeid","dayofmonth"], axis = 1)


        dataset["type"]=pd.Categorical(dataset['type'])
        dataset["type"]=dataset.type.cat.codes
        #pd.set_option('display.max_rows', dataset.shape[0]+1)

        #sys.exit()
        #plt.plot(dataset["dayofweek"])
        #plt.show()

        AI.actTrain(dataset,10)
    else:
        print("loading pretrained actTrain")


    if not os.path.isdir("models/probDistance"):
        #distance train
        data=pd.read_csv("data/PARSED/distancedtraindata.csv")
        print(data)
        #filter useful columns
        data=data.drop(['name','lat','lon'], axis = 1)
        print(data.dtypes)

        data["type"]=pd.Categorical(data['type'])
        data["type"]=data.type.cat.codes
        #pd.set_option('display.max_rows', dataset.shape[0]+1)
        print(data)
        #sys.exit()
        #plt.plot(dataset["dayofweek"])
        #plt.show()

        AI.distanceTrain(data)
    else:
        print("loading pretrained distanceTrain")


    #predict
    historicRaw=loadHistory()
    print("historicRaw",historicRaw)
    """

    data=pd.read_csv("data/PARSED/acttypetraindata.csv")
    #data=data.drop(['name',"dayofmonth","lat","lon"], axis = 1)
    print(data.tail(AI.lookbackLSTM))
    blocks=AI.predictBlocks(data,3)

    print("")
    print("------------------------------------------------------")
    print("")
    for b in blocks:
        print(b)
    #print(blocks["name"])
    """
