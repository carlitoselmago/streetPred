import numpy as np
import osmnx as ox
import networkx as nx
import pickle
import os.path

import pandas as pd
ox.config(log_console=True, use_cache=True)
import matplotlib.pyplot as plt

class locationProbAI():

    viewDistance=10000 #in meters
    minDistanceMatch=50 #in meters

    mS=1000.0 #multiplier size for node plotting size

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

    def timeStepAll(self,G):
        for i,row in enumerate(self.visitedLocations):
            row["visited"]-=self.timeStepPoint

            if row["visited"]>self.visitedMaxPoints:
                row["visited"]=self.visitedMaxPoints

            if row["visited"]<0.0:
                row["visited"]=0.0


    def assignVisitsToNodes(self,nodes,G):
        visits=[]

        visitedNodes=[]

        for i,l in enumerate(self.visitedLocations):
            closestNode=(ox.get_nearest_node(G, (l["lat"],l["lon"]),return_dist=True))
            osNodeId=str(closestNode[0])

            if int(closestNode[1])<self.minDistanceMatch:
                print("")
                self.visitedLocations[i]["nodeid"]=osNodeId
                visitedNodes.append(str(osNodeId))

        for i,node in nodes.iterrows():

            locationIndex=False

            try:
                locationIndex=self.find_index(self.visitedLocations,"nodeid",str(node["osmid"]))
                print("FOUND IN MAP! *************************************************************")
            except:
                pass
            if locationIndex:
                visits.append(self.visitedLocations[locationIndex]["visited"])
            else:
                visits.append(0.0)

        return visits


    def generateHeatMaps(self,df):

        for index, row in df.iterrows():
            print("")
            print("LOCATION ROW ::::::::::::::::::::::::::::: #",index)
            print("")
            if not any(d['placeid'] == row["placeid"] for d in self.visitedLocations):

                tempRow=row
                tempRow["visited"]=0.0
                self.visitedLocations.append(tempRow)

            vi=self.find_index(self.visitedLocations,"placeid",row["placeid"])
            self.visitedLocations[vi]["visited"]+=self.visitPoints


            location_point=(row["lat"],row["lon"])
            distance=self.viewDistance
            G = self.getMap(location_point,distance)
            #G = ox.graph_from_point(location_point, network_type='drive', dist=distance, simplify=False)

            self.timeStepAll(G)

            #Make geodataframes from graph data
            nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

            #Now make the same graph, but this time from the geodataframes
            #This will help retain the 'visits' columns
            nodes['visits']=self.assignVisitsToNodes(nodes,G)
            nodes['visitsS']=self.mS * nodes['visits']

            #print(self.visitedLocations)

            #G = ox.save_load.gdfs_to_graph(nodes, edges)
            G = ox.graph_from_gdfs(nodes, edges)

            #fig, ax =ox.plot_graph(G,show=True, close=False)
            #ox.plot_graph(G)


            if index>1:

                #ox.plot_graph(G,fig_height=8,fig_width=8,node_size=nodes['visits'], node_color=nc)
                #nc = ox.plot.get_node_colors_by_attr(G,'visits',cmap='plasma')
                #nc = ox.plot.get_node_colors_by_attr(G, 'visits',start=0.0,stop=self.visitedMaxPoints, cmap='plasma')
                print("nodes['visits']",nodes[nodes.visits > 0.00].to_string())
                nc = ox.plot.get_node_colors_by_attr(G, 'visits', cmap='plasma')
                fig, ax =ox.plot_graph(G, node_color=nc, node_size=nodes['visitsS'], edge_color='#333333', bgcolor='k',show=False)
                fig.savefig('data/MAPS/test_'+str(index)+'.png')

    def getMap(self,location,distance):
        filepath="data/cachedmaps/"+str(location[0])+'_'+str(location[1])+'-'+str(distance)+'.pkl'
        if os.path.isfile(filepath):
            print("loading cached map")
            #load
            with open(filepath, 'rb') as handle:
                b = pickle.load(handle)
            return b

        #call API and save
        G=ox.graph_from_point(location, network_type='drive', dist=distance, simplify=False)
        with open(filepath, 'wb') as handle:
            pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return G

if __name__ == "__main__":

    locAI = locationProbAI()

    #data=pd.read_csv("data/PARSED/traindata.csv")
    data=pd.read_csv("data/PARSED/acttypetraindata.csv")

    locAI.generateHeatMaps(data)
