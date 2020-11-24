# coding: utf-8
import json
from datetime import datetime
import csv
import glob
import sys

files='data/2017_MARCH.json'
maxp=100

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

for i,p in enumerate(datas):
    if "placeVisit" in p:

        p=p["placeVisit"]
        placeID=p["location"]["placeId"]
        if placeID not in places:
            places[placeID]=p["location"]
        placeName=None
        if "name" in p["location"]:
            placeName=p["location"]["name"]
        fechaTiempo=datetime.fromtimestamp(int(p["duration"]["startTimestampMs"])/1000.0)
        duration=float(( float(p["duration"]["endTimestampMs"]) - float(p["duration"]["startTimestampMs"]))// 3600000)
        fecha=fechaTiempo.date()
        if day<fecha:
            delta = fecha - day
            diasentre=(delta.days)

            print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
            if diasentre>1:
                if diasentre<10:
                    for dias in range(diasentre):
                        print("STAYHOMA!")
                        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
            day=fecha
        if duration>0.001:
            print(fechaTiempo,placeName)
            print("duracion",str(duration)+"h")
            print("")
    if "activitySegment" in p:
        #transito
        print(p["activitySegment"]["activityType"],"---------------------------------------------->")

        count+=1
        if count>maxp:
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
print(placesCSV)

f = open("data/history.csv", "w",encoding='utf-8',newline='')
writer = csv.DictWriter(
    f, fieldnames=["id", "latitudeE7","longitudeE7","address","name"])
writer.writeheader()
writer.writerows(placesCSV)
f.close()
