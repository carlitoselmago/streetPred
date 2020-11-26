import osmnx as ox
import matplotlib.pyplot as plt

#G = ox.graph_from_address('350 5th Ave, New York, New York', network_type='drive')
#ox.plot_graph(G)

#north, east, south, west = 33.798, -84.378, 33.763, -84.422
# Downloading the map as a graph object
#G =ox.graph_from_address('Plaça de Sants, Barcelona', network_type='walk')
location_point=(41.372925, 2.121151)
G = ox.graph_from_point(location_point, network_type='walk', dist=500, simplify=True)


areaX=[location_point[0],location_point[0]+1]
areaY=[location_point[1],location_point[1]+1]
nearestnodes=ox.get_nearest_nodes(G, areaX,areaY)#ox.get_nearest_nodes(G, (location_point))
print(nearestnodes)
"""
origin_point = (33.787201, -84.405076)
origin_node = ox.get_nearest_node(G, origin_point)
print("origin_node",origin_node)

ox.plot_graph(G)
for n in list(G.nodes(data=True)):
    print ("node",n)
"""

fig, ax =ox.plot_graph(G,show=False, close=False)
ax.scatter(location_point[1],location_point[0], c='red')
for n in nearestnodes:
    print(G.nodes[n]['x'])
    print(G.nodes[n]['y'])
    ax.scatter(G.nodes[n]['x'],G.nodes[n]['y'], c='green')
plt.show()
