import sys
import googlemaps
from datetime import datetime

try:
    from config import *
except:
    print("No existe el archivo config, stop")
    sys.exit()
from helpers import helpers
H=helpers()


def groute(start,end,timet,mode="walking"):
    gmaps = googlemaps.Client(key=config.gmapsAPI)

    # Geocoding an address
    #geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

    # Look up an address with reverse geocoding
    #reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

    # Request directions via public transit

    directions_result = gmaps.directions(start,#string coords separated by comma without spaces
                                         end,#string coords separated by comma without spaces
                                         #language="en",
                                         mode=mode,#options: driving, walking, bicycling, transit (metro etc)
                                         departure_time=timet)
    #print("directions_result",directions_result)
    return directions_result

    ######

    print(directions_result)
    routeSteps=[]
    for l in directions_result[0]["legs"][0]["steps"]:
        #}print(l["html_instructions"])
        routeSteps.append(l["html_instructions"])
        if "steps" in l:
            for i in l["steps"]:
                routeSteps.append(i["html_instructions"])
                #print(">>>>>>>",H.stripTags(i["html_instructions"]))
                #print(l)
        #print("")
    #print(directions_result[0]["legs"])
    return routeSteps



if __name__ == "__main__":
    now = datetime.now()
    route=groute("42.3441206,-3.7297982","41.3617417,2.0952498",now,"transit")
    print(route)

#graphic representation:
#https://stackoverflow.com/questions/16180104/get-a-polyline-from-google-maps-directions-v3
