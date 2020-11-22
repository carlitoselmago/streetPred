from OSMPythonTools.api import Api
from OSMPythonTools.overpass import Overpass
from pprint import pprint
overpass = Overpass()

around=100#around in meters
loc=[41.3735485,2.1215157]

result = overpass.query('node["amenity"](around:'+str(around)+','+str(loc[0])+','+str(loc[1])+'); out;')
print(result.toJSON())
for n in result.toJSON()["elements"]:
    #print(n.tags())
    elem={"id":n["id"],"type":n["tags"]["amenity"]}
    if "name" in n["tags"]:
        elem["name"]=n["tags"]["name"]
    else:
        elem["name"]=None
    elem["loc"]=[n["lat"],n["lon"]]

    #print(," - ",n["tags"]["name"],n["lat"])
    print(elem)

#mas info sobre el objeto result https://github.com/mocnik-science/osm-python-tools/blob/master/docs/overpass.md


"""
api = Api()
overpass_query = 'node(57.7,11.9,57.8,12.0)[amenity=bar];out;'
loc = api.query(overpass_query )
print(loc)
"""
