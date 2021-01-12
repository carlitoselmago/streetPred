import pandas as pd
import os

from googleHistoryParser import parseHistory,loadHistory
from probAI import probAI

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
    dataset=data.drop(['name', 'duration',"placeid","lat","lon","lasttransport"], axis = 1)
    #dataset=data.drop(['name', 'duration',"placeid","dayofmonth"], axis = 1)


    dataset["type"]=pd.Categorical(dataset['type'])
    dataset["type"]=dataset.type.cat.codes
    #pd.set_option('display.max_rows', dataset.shape[0]+1)

    #sys.exit()
    #plt.plot(dataset["dayofweek"])
    #plt.show()

    AI.actTrain(dataset,100)
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



data=pd.read_csv("data/PARSED/acttypetraindata.csv")
#data=data.drop(['name',"dayofmonth","lat","lon"], axis = 1)

blocks=AI.predictBlocks(data,12)

print("")
print("------------------------------------------------------")
print("")
for b in blocks:
    print(b)
#print(blocks["name"])
