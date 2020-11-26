import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

import networkx as nx
import osmnx as ox
ox.config(use_cache=True, log_console=True)

def create_graph(loc, dist, transport_mode, loc_type="address"):
    """Transport mode = ‘walk’, ‘bike’, ‘drive’, ‘drive_service’, ‘all’, ‘all_private’, ‘none’"""
    if loc_type == "address":
            G = ox.graph_from_address(loc, dist=dist, network_type=transport_mode)
    elif loc_type == "points":
            G = ox.graph_from_point(loc, dist=dist, network_type=transport_mode )
    return G

def coord2Node(G,coord):
    return ox.get_nearest_node(G, (coord[1],coord[0]))

G = create_graph("Barcelona", 6500, "walk")
#ox.plot_graph(G)

#G = ox.add_edge_speeds(G) #Impute
#G = ox.add_edge_travel_times(G) #Travel time

# Calculate the shortest path
#route = nx.shortest_path(G, start_node, end_node, weight='travel_time')
coords=[
[2.1702181413650727,41.38770508700309],[2.1702181413650727,41.38770508700309],[2.1695049,41.3886936],[2.1692622,41.3888644],[2.1690645,41.3890507],[2.1689113,41.3892197],[2.1687414,41.3893728],[2.1687557,41.3894979],[2.1686018,41.3896605],[2.1684734,41.3897762],[2.1678293,41.3901778],[2.1677146,41.390274],[2.1675111,41.3903827],[2.1673755,41.3906692],[2.1671064,41.390771],[2.1669262,41.3908686],[2.1667965,41.3910065],[2.1665507,41.3911726],[2.1663285,41.3913862],[2.1662142,41.3915669],[2.1660237,41.3917971],[2.1657899,41.3919498],[2.1657318,41.3919621],[2.1656546,41.3925826],[2.1669494,41.3937463],[2.1684794,41.3948974],[2.1681569,41.3953695],[2.1677458,41.3955231],[2.1678275,41.3955222],[2.1679183635330292,41.39546825150053],[2.1679183635330292,41.39546825150053]
]
route=[]
for c in coords:
    p=coord2Node(G,c)
    if p not in route:
        route.append(p)
print(route)
#route=[145173356, 299734358, 5061966869, 301823631, 145181102, 299823745, 60902851, 60902873, 60902861, 60902887, 1374702846, 1739118998, 60906511, 60911679, 7373668, 60911678, 2487024335, 2487024331, 60786992, 1374678095, 7373669, 60890012, 25587532, 1375971951, 314452681]

#Plot the route and street networks
#ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k' );
ox.plot_route_folium(G,route)
