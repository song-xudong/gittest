# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 17:57:03 2022

作者：李一邨
人工智能算法案例大全：基于Python
浙大城市学院、杭州伊园科技有限公司
浙江大学 博士
中国民盟盟员
Email:liyicun_yykj@163.com
"""


#%%


import networkx as nx
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')


#%%


import networkx as nx
import matplotlib.pyplot as plt

G_symmetric = nx.Graph()

G_symmetric.add_edge('Laura', 'Steven')
G_symmetric.add_edge('Steven',  'John')
G_symmetric.add_edge('Steven',  'Michelle')
G_symmetric.add_edge('Laura',   'Michelle')
G_symmetric.add_edge('Michelle','Marc')
G_symmetric.add_edge('George',  'John')
G_symmetric.add_edge('George',  'Steven')
G_symmetric.add_edge('Quan',  'John')
print(nx.info(G_symmetric))

#%%


plt.figure(figsize=(10,10))
nx.draw_networkx(G_symmetric);

#%%

G_asymmetric = nx.DiGraph()
G_asymmetric.add_edge('A','B')
G_asymmetric.add_edge('B','A')

G_asymmetric.add_edge('A','D')
G_asymmetric.add_edge('C','A')
G_asymmetric.add_edge('D','E')

G_asymmetric.add_edge('F','K')
G_asymmetric.add_edge('B','F')

nx.spring_layout(G_asymmetric)
nx.draw_networkx(G_asymmetric)

#%%


nx.spring_layout(G_asymmetric)
nx.draw_networkx(G_asymmetric)

#%%

G_weighted = nx.Graph()


G_weighted.add_edge('Steven',  'Laura',   weight=2)
G_weighted.add_edge('Steven',  'Marc',    weight=8)
G_weighted.add_edge('Steven',  'John',    weight=11)
G_weighted.add_edge('Steven',  'Michelle',weight=1)
G_weighted.add_edge('Laura',   'Michelle',weight=1)
G_weighted.add_edge('Michelle','Marc',    weight=1)
G_weighted.add_edge('George',  'John',    weight=8)
G_weighted.add_edge('George',  'Steven',  weight=4)


elarge = [(u, v) for (u, v, d) in G_weighted.edges(data=True) if d['weight'] > 8]
esmall = [(u, v) for (u, v, d) in G_weighted.edges(data=True) if d['weight'] <= 8]
print('Large Edges: ', elarge)
print('Small Edges: ', esmall)

pos = nx.circular_layout(G_weighted)  

nx.draw_networkx_nodes(G_weighted, pos, node_size=700)


nx.draw_networkx_edges(G_weighted, pos, edgelist=elarge,width=6)
nx.draw_networkx_edges(G_weighted, pos, edgelist=esmall,width=2, alpha=0.5, edge_color='b', style='dashed')


nx.draw_networkx_labels(G_weighted, pos, font_size=20, font_family='sans-serif')

plt.axis('off')
plt.show()

#%%


print('Michelle: ', round(nx.clustering(G_symmetric,'Michelle'),3))


print('Laura: ', round(nx.clustering(G_symmetric,'Laura'),3))

#%%



round(nx.average_clustering(G_symmetric),3)

#%%


nx.degree(G_symmetric, 'Michelle')



#%%


nx.shortest_path(G_symmetric, 'Michelle', 'John')

nx.shortest_path_length(G_symmetric, 'Michelle', 'John')

#%%


S = nx.bfs_tree(G_symmetric, 'Michelle')
nx.draw_networkx(S)

#%%


nx.eccentricity(G_symmetric,'Michelle')
nx.eccentricity(G_symmetric,'Steven')

#%%


Degree_Centrality = nx.degree_centrality(G_symmetric)
for key in Degree_Centrality:
    print(key,":", round(Degree_Centrality[key],3))
    
#%%


Eigen_cent = nx.eigenvector_centrality(G_symmetric)
for key in Eigen_cent:
    print(key,":", round(Eigen_cent[key],3))

#%%


Close_cent = nx.closeness_centrality(G_symmetric)
for key in Close_cent :
    print(key,":", round(Close_cent [key],3))
    
#%%
   

nx.betweenness_centrality(G_symmetric)

#%%
pos = nx.spring_layout(G_symmetric)
betCent = nx.betweenness_centrality(G_symmetric, normalized=True, endpoints=True)
node_color = [2000 * G_symmetric.degree(v) for v in G_symmetric]
node_size =  [v * 1000 for v in betCent.values()]
plt.figure(figsize=(7,6))
nx.draw_networkx(G_symmetric, pos=pos, with_labels=True,
                 node_color=node_color,
                 node_size=node_size )
plt.axis('off');
#%%
sorted(betCent, key=betCent.get, reverse=True)[:5]

#%%





import pandas as pd

df = pd.read_csv('network_combined.txt')
df.info()
df.head()
#%%
G_fb = nx.read_edgelist("network_combined.txt", create_using = nx.Graph(), nodetype=int)
print(nx.info(G_fb))
#%%

plt.figure(figsize=(20,20))
nx.draw_networkx(G_fb);


#%%



pos = nx.spring_layout(G_fb)
betCent = nx.betweenness_centrality(G_fb, normalized=True, endpoints=True)
node_color = [20000.0 * G_fb.degree(v) for v in G_fb]
node_size =  [v * 10000 for v in betCent.values()]
plt.figure(figsize=(20,20))
nx.draw_networkx(G_fb, pos=pos, with_labels=False,
                 node_color=node_color,
                 node_size=node_size )
plt.axis('off');
#%%



sorted(betCent, key=betCent.get, reverse=True)[:5]


betCent = nx.betweenness_centrality(G_fb, normalized=True, endpoints=True)
Close_cent = nx.closeness_centrality(G_fb)
Eigen_cent = nx.eigenvector_centrality(G_fb)
Degree_Centrality = nx.degree_centrality(G_fb)

sorted(betCent, key=betCent.get, reverse=True)[:5]
sorted(Close_cent, key=Close_cent.get, reverse=True)[:5]
sorted(Eigen_cent, key=Eigen_cent.get, reverse=True)[:5]
sorted(Degree_Centrality, key=Degree_Centrality.get, reverse=True)[:5]



#%%




import csv                                                             
from operator import itemgetter
 

with open('quakers_nodelist.csv', 'r') as nodecsv:                 
    nodereader = csv.reader(nodecsv)                                       
    nodes = [n for n in nodereader][1:]                                    


node_names = [n[0] for n in nodes]                                       


with open('quakers_edgelist.csv', 'r') as edgecsv:                         
    edgereader = csv.reader(edgecsv)                                   
    edges = [tuple(e) for e in edgereader][1:]                         


G = nx.Graph(name="Quakers Social Network")     

G.add_nodes_from(node_names)  

G.add_edges_from(edges)

print(nx.info(G)) 

#%%

plt.figure(figsize=(15,15))
nx.draw_networkx(G);

#%%


hist_sig_dict = {}
gender_dict = {}
birth_dict = {}
death_dict = {}
id_dict = {}

for node in nodes:
    hist_sig_dict[node[0]] = node[1] 
    gender_dict[node[0]] = node[2]
    birth_dict[node[0]] = node[3]
    death_dict[node[0]] = node[4]
    id_dict[node[0]] = node[5]


nx.set_node_attributes(G, hist_sig_dict, 'historical_significance')
nx.set_node_attributes(G, gender_dict, 'gender')
nx.set_node_attributes(G, birth_dict, 'birth_year')
nx.set_node_attributes(G, death_dict, 'death_year')
nx.set_node_attributes(G, id_dict, 'sdfb_id')


for n in G.nodes():
    print(n, G.nodes[n]['birth_year'])
    
    
#%%



density = nx.density(G)
print("Network density:", round(density,3))

#%%



degree_dict = dict(G.degree(G.nodes()))
nx.set_node_attributes(G, degree_dict, 'degree')


print(G.nodes['William Penn'],'\n')

#%%


sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)
print("Top 20 nodes by degree:")
for d in sorted_degree[:20]:
    print(d)

#%%


betweenness_dict = nx.betweenness_centrality(G) 

eigenvector_dict = nx.eigenvector_centrality(G) 


nx.set_node_attributes(G, betweenness_dict, 'betweenness')
nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')


sorted_betweenness = sorted(betweenness_dict.items(), 
                            key=itemgetter(1), reverse=True)

print("Top 20 nodes by betweenness centrality:")
for b in sorted_betweenness[:20]:
    print(b[0],":",round(b[1],3))

#%%


top_betweenness = sorted_betweenness[:20]


for tb in top_betweenness: # Loop through top_betweenness
    degree = degree_dict[tb[0]] # Use degree_dict to access a node's degree, see footnote 2
    print("Name:", tb[0], "| Betweenness Centrality:", round(tb[1],2), "| Degree:", degree)
