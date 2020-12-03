import numpy as np
import osmnx as ox
import pandas as pd
ox.config(log_console=True, use_cache=True)
import matplotlib.pyplot as plt

class locationProbAI():

    visitedLocations=[] #list of locations with id, lat, lon and value of visit from 0.0 to 1.0
    visitedMaxPoints=1.0
    visitPoints=0.2
    timeStepPoint=0.002

    def __init__(self):
        pass

    def find_index(self,dicts, key, value):
        class Null: pass
        for i, d in enumerate(dicts):
            if d.get(key, Null) == value:
                return i
        else:
            raise ValueError('no dict with the key and value combination found')

    def timeStepAll(self):
        for i,row in enumerate(self.visitedLocations):
            row["visited"]-=self.timeStepPoint
            if row["visited"]>self.visitedMaxPoints:
                row["visited"]=self.visitedMaxPoints
            if row["visited"]<0.0:
                row["visited"]=0.0

    def generateHeatMaps(self,df):

        for index, row in df.iterrows():

            if not any(d['placeid'] == row["placeid"] for d in self.visitedLocations):

                tempRow=row
                tempRow["visited"]=0.0
                self.visitedLocations.append(tempRow)

            vi=self.find_index(self.visitedLocations,"placeid",row["placeid"])
            self.visitedLocations[vi]["visited"]+=self.visitPoints

            self.timeStepAll()

        print(self.visitedLocations)
        sys.exit()
        #G = ox.graph_from_address('350 5th Ave, New York, New York', network_type='drive')
        #ox.plot_graph(G)

        #north, east, south, west = 33.798, -84.378, 33.763, -84.422
        # Downloading the map as a graph object
        #G =ox.graph_from_address('Plaça de Sants, Barcelona', network_type='walk')
        location_point=(41.372925, 2.121151)
        distance=500
        G = ox.graph_from_point(location_point, network_type='walk', dist=distance, simplify=True)


        fig, ax =ox.plot_graph(G,show=False, close=False)
        ox.plot_graph(G)


if __name__ == "__main__":

    locAI = locationProbAI()

    data=pd.read_csv("data/PARSED/traindata.csv")

    locAI.generateHeatMaps(data)



"""


#using https://github.com/gboeing/osmnx-examples/blob/master/notebooks/12-node-elevations-edge-grades.ipynb
# + https://stackoverflow.com/questions/62817760/coloring-nodes-according-to-an-attribute-in-osmnx

place = 'San Francisco'
place_query = {'city':'Barcelona', 'country':'Spain'}
G = ox.graph_from_place(place_query, network_type='drive')



# get one color for each node, by elevation, then plot the network
nc = ox.plot.get_node_colors_by_attr(G, 'elevation', cmap='plasma')
fig, ax = ox.plot_graph(G, node_color=nc, node_size=5, edge_color='#333333', bgcolor='k')
"""
