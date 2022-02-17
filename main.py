import os
import configparser
import pandas as pd

from googleHistoryParser import loadHistory, parseHistory
from probAI import probAI

config = configparser.ConfigParser()
config.read('settings.ini')
AI=probAI()

#history=parseHistory()
try:
    os.remove("data/PARSED/acttypetraindata.csv")
except:
    pass
parseHistory()
#print("historicRaw",historicRaw)


if not os.path.isdir("models/probAct"):
    data=pd.read_csv("data/PARSED/acttypetraindata.csv")

    #filter useful columns
    dataset=data.drop(['name', 'duration',"placeid","lat","lon","lasttransport","year"], axis = 1)
    #dataset=data.drop(['name', 'duration',"placeid","dayofmonth"], axis = 1)


    dataset["type"]=pd.Categorical(dataset['type'])
    dataset["type"]=dataset.type.cat.codes
    #pd.set_option('display.max_rows', dataset.shape[0]+1)

    #sys.exit()
    #plt.plot(dataset["dayofweek"])
    #plt.show()
    print(dataset)
    AI.actTrain(dataset,100)
else:

    print("loading pretrained actTrain")


if not os.path.isdir("models/probDistance"):
    #distance train
    data=pd.read_csv("data/PARSED/distancedtraindata.csv")
    #print(data)
    #filter useful columns
    data=data.drop(['name','lat','lon',"lasttransport"], axis = 1)
    #print(data.dtypes)

    data["type"]=pd.Categorical(data['type'])
    data["type"]=data.type.cat.codes
    #pd.set_option('display.max_rows', dataset.shape[0]+1)
    #print(data)
    #sys.exit()
    #plt.plot(dataset["dayofweek"])
    #plt.show()

    AI.distanceTrain(data)
else:
    print("loading pretrained distanceTrain")



data=pd.read_csv("data/PARSED/acttypetraindata.csv")

blocks=AI.predictBlocks(data,config["base"]) 

print("")
print("------------------------------------------------------")
print("")
for b in blocks:
    print(b)
#print(blocks["name"])
