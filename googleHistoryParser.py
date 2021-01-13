# coding: utf-8
import json
from datetime import datetime
import csv
import glob
import sys
import pandas as pd
import numpy as np
import re

from helpers import helpers

"""
HYPERPARAMETERS:
"""
_DAYSPLIT=6 #hour blocks to split 24h of each day

"""
end HYPERPARAMETERS:
"""

H=helpers()

limit=0

#load cats from google sheet
catLocationsURL='https://docs.google.com/spreadsheets/d/e/2PACX-1vShw1idYHeOZpKT0w10zHVUkbBEQU_a7E4w2qlBSPN7hNr3QpV2M_r5L3Bs3_Jw5_AlvSpBx8GGuoyP/pub?output=csv'
catLocations = pd.read_csv(catLocationsURL)

def sortkey(x):
    monthIndex=["JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE","JULY","AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER"]
    x=x.replace("data/ALL/","")
    parts = re.split('[-._]', x)
    return [int(parts[0]), monthIndex.index((parts[1]))]

def loadHistory():

    files=[]
    for name in glob.glob('data/ALL/*.json'):
        sortkey(name)
        files.append(name)
    files=sorted(files, key=sortkey)
    datas=[]
    for name in files:
        #print(name)
        with open(name,mode="r", encoding="utf-8") as json_file:
            data = json.load(json_file)['timelineObjects']
            datas+=data

    print("TOTAL timelineobjeects",len(datas))
    return datas



def parseHistory():

    # JOIN ALL FILES DATA
    datas=loadHistory()

    # END JOIN

    places={}

    count=0
    day=datetime.fromtimestamp(0).date()
    fecha=False

    dayBlocks= [None] * _DAYSPLIT
    lastBlock=False

    defaultActivityType="Stu"

    TRAINBLOCKS=[]
    travelMoves=[]

    for i,p in enumerate(datas):

        if fecha:
            if day<fecha:
                delta = fecha - day
                diasentre=(delta.days)
                print("")
                print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
                print("")

                #DAY BLOCK :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                dayBlocks=H.spreadBlocks(dayBlocks,lastBlock)
                print("¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿")
                print ("dayBlocks",dayBlocks)
                print("¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿")
                print("")

                if H.validDayblocks(dayBlocks):
                    TRAINBLOCKS.append(dayBlocks)

                lastBlock=dayBlocks[-1]
                #reset dayblock
                dayBlocks= [None] * _DAYSPLIT

                if diasentre>0:
                    if diasentre<10:
                        for dias in range(diasentre):

                            #ADD STAY HOME BLOCK

                            emptyDay=[None] * _DAYSPLIT
                            for ds in range(_DAYSPLIT):
                                lastAct=lastBlock#TRAINBLOCKS[-1][-1]
                                if ds==0:
                                    #first block
                                    lastAct["dayofweek"]=lastAct["dayofweek"]+1
                                    lastAct["dayofmonth"]=lastAct["dayofmonth"]+1
                                lastAct["timeblock"]=ds
                                lastAct["name"]=lastAct["name"]#"::::REST DAY::::"
                                emptyDay[ds]=lastAct
                            TRAINBLOCKS.append(emptyDay)

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

                actT=None#defaultActivityType

                actTDB=catLocations.loc[catLocations["id"]==placeID]["cat"].values[0]
                if not pd.isnull(actTDB):
                    actT=actTDB

                dayofweek=fechaTiempo.weekday()
                month=fechaTiempo.month
                year=fechaTiempo.year
                dayofmonth=fechaTiempo.day

                activity={"type":actT,"duration":duration,"name":placeName,"placeid":placeID,"dayofmonth":dayofmonth,"dayofweek":dayofweek,"month":month,"year":year,"lat":p["location"]["latitudeE7"]/ 1e7,"lon":p["location"]["longitudeE7"]/ 1e7,"lasttransport":H.selectTransportMode(travelMoves)}

                travelMoves=[]

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
                travelMoves.append(p["activitySegment"]["activityType"])
            except:
                pass

            if limit>0:
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

    f = open("data/PARSED/placeshistory.csv", "w",encoding='utf-8',newline='')
    writer = csv.DictWriter(
        f, fieldnames=["id", "latitudeE7","longitudeE7","address","name"])
    writer.writeheader()
    writer.writerows(placesCSV)
    f.close()


    ##### WRITE TRAIN DATA
    print("")
    print("END:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    INDBLOCKS=[]

    for tb in TRAINBLOCKS:
        for i,b in enumerate(tb):
            bb=b
            bb["timeblock"]=i
            if bb["type"]!="-":

                if len(INDBLOCKS)>2:
                    if bb["timeblock"]==0 and bb["dayofweek"]==INDBLOCKS[i-1]["dayofweek"]:
                        bb["dayofweek"]+=1

                INDBLOCKS.append(bb)
                #print (bb)

    print("")
    print("TRAINBLOCKS")
    df = pd.DataFrame(INDBLOCKS)

    #drop rows wich dont have a type category
    df.dropna(subset=['type'], inplace=True)

    print(df)
    df.to_csv('data/PARSED/acttypetraindata.csv',index=False)
    return df

if __name__ == "__main__":
    parseHistory()
