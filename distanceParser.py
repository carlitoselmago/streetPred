import numpy as np
import pickle
import os.path
import pandas as pd
import geopy.distance

class distanceProbAI():

    dataDistancedFile='data/PARSED/distancedtraindata.csv'
    maxKm=1500.0 #max distance traveled
    maxSmallKm=40.0 #average local max distance

    def __init__(self):
        pass

    def parseData(self,df):

        #if os.path.isfile(self.dataDistancedFile):
        #    return pd.read_csv(self.dataDistancedFile)

        #else
        lastLoc=(df["lat"].values[0],df["lon"].values[0])

        distancedData=[]

        for index, row in df.iterrows():
            currentLoc=(row["lat"],row["lon"])
            distance=self.calcDistance(currentLoc,lastLoc)


            newRow={"type":row["type"],"name":row["name"],"dayofmonth":row["dayofmonth"],"dayofweek":row["dayofweek"],"month":row["month"],"year":row["year"],"lat":row["lat"],"lon":row["lon"],"timeblock":row["timeblock"]}
            newRow["distancefromlast"]=distance #in KM float

            if row["type"]=="Portal":
                newRow["distancefromlast"]=(newRow["distancefromlast"]*self.maxSmallKm)/self.maxKm

            distancedData.append(newRow)

            ################
            lastLoc=currentLoc

        #save data
        newdf = pd.DataFrame(distancedData)
        newdf.to_csv(self.dataDistancedFile, index=False)
        return newdf

    def trainDistances(self,df):
        trainData=self.parseData(df)



    def calcDistance(self,loc1,loc2):
        return geopy.distance.distance(loc1, loc2).km

if __name__ == "__main__":

    locAI = distanceProbAI()

    data=pd.read_csv("data/PARSED/acttypetraindata.csv")

    locAI.trainDistances(data)
