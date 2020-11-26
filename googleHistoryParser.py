# coding: utf-8
import json
from datetime import datetime
import csv
import glob
import sys
import pandas as pd

from helpers import helpers

"""
HYPERPARAMETERS:
"""
_DAYSPLIT=6 #hour blocks to split 24h of each day

"""
end HYPERPARAMETERS:
"""

H=helpers()

limit=100

#load cats from google sheet
catLocationsURL='https://docs.google.com/spreadsheets/d/e/2PACX-1vShw1idYHeOZpKT0w10zHVUkbBEQU_a7E4w2qlBSPN7hNr3QpV2M_r5L3Bs3_Jw5_AlvSpBx8GGuoyP/pub?output=csv'
catLocations = pd.read_csv(catLocationsURL)



# JOIN ALL FILES DATA
datas=[]
for name in glob.glob('data/ALL/*.json'):
    print(name)
    with open(name,mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)['timelineObjects']
        datas+=data


print("TOTAL timelineobjeects",len(datas))

# END JOIN

places={}

count=0
day=datetime.fromtimestamp(0).date()
fecha=False

dayBlocks= [None] * _DAYSPLIT

defaultActivityType="Stu"

for i,p in enumerate(datas):

    if fecha:
        if day<fecha:
            delta = fecha - day
            diasentre=(delta.days)
            print("")
            print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
            print("")

            print("¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿")
            print ("dayBlocks",dayBlocks)
            print("¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿")
            print("")

            #reset dayblock
            dayBlocks= [None] * _DAYSPLIT

            if diasentre>1:
                if diasentre<10:
                    for dias in range(diasentre):

                        print("STAYHOMA!")
                        print("")
                        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                        print("")

        day=fecha

    if "placeVisit" in p:

        p=p["placeVisit"]
        placeID=p["location"]["placeId"]
        if placeID not in places:
            places[placeID]=p["location"]
        placeName=None
        if "name" in p["location"]:
            placeName=p["location"]["name"]

        duration=float(( float(p["duration"]["endTimestampMs"]) - float(p["duration"]["startTimestampMs"]))// 3600000)
        fechaTiempo=datetime.fromtimestamp(int(p["duration"]["startTimestampMs"])/1000.0)
        fecha=fechaTiempo.date()



        if duration>0.001:
            print(fechaTiempo,placeName)
            print("duracion",str(duration)+"h")

            daysplitIndex=H.timeBlockIndex(_DAYSPLIT,float(p["duration"]["startTimestampMs"]),float(p["duration"]["endTimestampMs"]))

            print("daysplitIndex",daysplitIndex,"%%%%%%%%%%%%%%%%%%")
            print("")

            actT=defaultActivityType

            activity={"type":actT,"duration":duration,"name":placeName}

            try:
                if duration>dayBlocks[daysplitIndex]["duration"]:
                    dayBlocks[daysplitIndex]=activity
            except:

                dayBlocks[daysplitIndex]=activity

    if "activitySegment" in p:
        #transito

        fechaTiempo=datetime.fromtimestamp(int(p["activitySegment"]["duration"]["startTimestampMs"])/1000.0)
        fecha=fechaTiempo.date()
        try:
            print(p["activitySegment"]["activityType"],"---------------------------------------------->")
        except:
            pass

        if i>limit:
            break

#print(places)
placesCSV=[]
for id in places:
    place=places[id]
    if "semanticType" in place:
        del place["semanticType"]
    if "sourceInfo" in place:
        del place["sourceInfo"]
    del place["locationConfidence"]
    del place["placeId"]
    if "address" in place:
        place["address"]=place["address"].replace("\n", " ")
    else:
        place["address"]=""
    place["id"]=id
    placesCSV.append(place)
#print(placesCSV)

f = open("data/history.csv", "w",encoding='utf-8',newline='')
writer = csv.DictWriter(
    f, fieldnames=["id", "latitudeE7","longitudeE7","address","name"])
writer.writeheader()
writer.writerows(placesCSV)
f.close()
