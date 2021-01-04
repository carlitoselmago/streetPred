import googlemaps
from datetime import datetime
import config
from helpers import helpers
H=helpers()


def groute(start,end,timet):
    gmaps = googlemaps.Client(key=config.gmapsAPI)

    # Geocoding an address
    #geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

    # Look up an address with reverse geocoding
    #reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

    # Request directions via public transit

    directions_result = gmaps.directions(start,#string coords separated by comma without spaces
                                         end,#string coords separated by comma without spaces
                                         language="es",
                                         mode='driving',#mode='transit'
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
    route=groute("41.3730711,2.1179825","41.3854471,2.1271785",now)
    print(route)

#graphic representation:
#https://stackoverflow.com/questions/16180104/get-a-polyline-from-google-maps-directions-v3
