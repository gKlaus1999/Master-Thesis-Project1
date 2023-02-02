import networkx as nx
import random
import statistics
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

#Function that shuffles the weights in a network
#Input: Network with weights
#Output: Network with mixed weights
def shuffleWeights(G):
    weights=getWeights(G)
    random.shuffle(weights)

    addWeights(weights,G)
    return(G)

#Funtion that gets the Weights of a network
#Input: Network
#Output: list with all the weights
def getWeights(R):
    weights=nx.get_edge_attributes(R,'weight')
    weights=list(weights.values())
    return(weights)

#Function that adds weights to a network
#Input: Weights and network
#Output: Network with weights
def addWeights(weights,G):
    mapping = {}
    edgelist=list(G.edges)
    mapping=dict(zip(edgelist,weights))
    nx.set_edge_attributes(G, values = mapping, name = 'weight')
    return(G)
 
#Function that adds invweights to a network
#Input: Weights and network
#Output: Network with inverted weights
def addInvWeights(G):
    weights=nx.get_edge_attributes(G,'weight')
    weights=list(weights.values())
    weights=[1/m for m in weights]
    mapping = {}
    edgelist=list(G.edges)
    mapping=dict(zip(edgelist,weights))
    nx.set_edge_attributes(G, values = mapping, name = 'invweight')
    return(G)

#Function that checks if two nodes are connected:
#Input: touple of nodes
#Output: True or False
def nodes_connected(G, u, v):
    return u in G.neighbors(v)

#Function that randomizes a network
#Input: Network
#Output: randomized network
def randomizer(F):
    for node in range(F.number_of_nodes()):
        neighbours=list(F[node].keys())
        for neigh in range(len(neighbours)):
            m=0
            switch=0
            while m <100 and not(switch):
                nodeneighs = [n for n in neighbours if n != neigh]
                if len(nodeneighs)==0:
                    break
                nodeneigh = random.choice(nodeneighs)

                neighneighs = list(F[nodeneigh].keys())
                neighneighs = [n for n in neighneighs if n != nodeneigh]
                if len(neighneighs)>0:
                    neighneigh = random.choice(neighneighs)
                
                    if nodes_connected(F, node, neighneigh)and not(nodes_connected(F, neigh, nodeneigh)):
                        F.remove_edge(node,neighneigh)
                        F.add_edge(neigh, nodeneigh)
                        switch=1
                    elif not(nodes_connected(F, node, neighneigh)) and nodes_connected(F, neigh, nodeneigh):
                        F.remove_edge(neigh, nodeneigh)
                        F.add_edge(node, neighneigh)
                        switch=1
                
                    else:
                        m+=1
                else:
                    m+=1
    return(F)

net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/agtanet.txt'
G = nx.read_weighted_edgelist(net,nodetype = int) #read in network


Rnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/Redglist.txt'
R = nx.read_weighted_edgelist(Rnet, nodetype = int)

Hadzanet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/CampHadza.txt'
G = nx.read_edgelist(Hadzanet, nodetype=int)

HadzaRnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/CampRHadza.txt'
R = nx.read_weighted_edgelist(HadzaRnet, nodetype=int)
'''
Hadzanet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/HoneyHadza.txt'
G = nx.read_weighted_edgelist(Hadzanet, nodetype=int)

HadzaRnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/HoneyRHadza.txt'
R = nx.read_weighted_edgelist(HadzaRnet, nodetype=int)
'''
print(nx.number_of_edges(G))
largest_cc = min(nx.connected_components(G), key=len)
#G=G.subgraph(largest_cc).copy()


quit()

nx.set_edge_attributes(G, values = 1, name = 'weight')

mapping = {}
nodelist=list(G.nodes)
for i in range(nx.number_of_nodes(G)):
    mapping[nodelist[i]]=i
G=nx.relabel_nodes(G, mapping) #relabel nodes 
R=nx.relabel_nodes(R, mapping)

pos = nx.spring_layout(G, seed=3113794652)
nx.draw_networkx_nodes(G,pos,node_color="red")
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.title("Hadza H Network")
plt.show()

RGs =[]
for i in range(100):
    net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaHRanNet/HadzaHRan{i}.txt'
    temp = nx.read_weighted_edgelist(net,nodetype = int) #read in networkmapping = {}
    nodelist=list(temp.nodes)
    for i in range(nx.number_of_nodes(temp)):
        mapping[nodelist[i]]=i
    temp=nx.relabel_nodes(temp, mapping) #relabel nodes 
    RGs.append(temp)


pos2 = nx.spring_layout(RGs[0], seed=3113794652)
nx.draw_networkx_nodes(RGs[0],pos2,node_color="Green")
nx.draw_networkx_edges(RGs[0],pos2)
nx.draw_networkx_labels(RGs[0],pos2)
plt.title("Random Hadza H Network")
plt.show()

#create comparable networks for the relationship network
Rdens=nx.density(R)
Rsize=R.number_of_nodes()
Redges=R.number_of_edges()
Rweights=getWeights(R)

#create comparable networks for the social network
dens=nx.density(G)
size=G.number_of_nodes()
edges=G.number_of_edges()
weights=getWeights(G)
print(size)
print(edges)
#BA = nx.barabasi_albert_graph(size,round(edges/size))
Ran=nx.gnm_random_graph(size,edges)
Ran=addWeights(weights,Ran)

#create comparable small world network
Smw = nx.watts_strogatz_graph(size, int(2*edges/size), 0.1)#int(2*edges/size)
Smw = addWeights(weights,Smw)
print(Smw.number_of_nodes())
print(Smw.number_of_edges())
#create scale free graph
Sf = nx.scale_free_graph(size)
Sf = addWeights(weights,Sf)

#G=addInvWeights(G)
networks = [G, Smw]
#Degree Distribution
totweights=list(G.degree(weight='weight'))

degs=[]
dens=[]
trans=[]
clusts=[]
eccs=[]
diams=[]
spls=[]

for network in networks:
    
    #degrees
    degs.append(list(network.degree()))

    #density
    dens.append(nx.density(network))

    #transitivity
    trans.append(nx.transitivity(network))

    #clusters
    clusts.append(nx.clustering(network))#, weight='weight'))

    #eccentricity
    #eccs.append(nx.eccentricity(network))

    #diameter
    #diams.append(nx.diameter(network))

    #Shortest path length
    #spls.append(nx.average_shortest_path_length(network))#weight='invweight'




degs100=[]
dens100=[]
trans100=[]
clusts100=[]
eccs100=[]
diams100=[]
spls100=[]
#print(spls)
for network in RGs:
    #degrees
    degs100.append(list(network.degree()))

    #density
    dens100.append(nx.density(network))

    #transitivity
    trans100.append(nx.transitivity(network))

    #clusters
    clusts100.append(nx.clustering(network))#, weight='weight'))

    #eccentricity
    #eccs100.append(nx.eccentricity(network))

    #diameter
    #diams100.append(nx.diameter(network))

    #Shortest path length
    #spls100.append(nx.average_shortest_path_length(network))#weight='invweight'

#Shortest path length
#spls.insert(1, statistics.mean(spls100))
#print(f"spls:{spls}")

#degs
temp=[]
for deg in degs100:
    temp.extend(deg)
degsran= [d[1] for d in temp]

#density
dens.insert(1,statistics.mean(dens100))
print(f"Density:{dens}")

#diameter
#diams.insert(1, statistics.mean(diams100))
#print(f"diameter:{diams}")

#clusters
temp=[list(clust.values()) for clust in clusts100]
clustran=[]
for clust in temp:
    clustran.extend(clust)
#print(clustran)

#eccentricity
temp=[list(ecc.values())for ecc in eccs100]
eccsran=[]
#for ecc in temp:
 #   eccsran.extend(ecc)
#print(eccsran)

#Transitivity
trans.insert(1,statistics.mean(trans100))
print(f"transitivity={trans}")

'''

temp= list(eccs[0].values())
temp2=list(eccsran)
temp3=list(eccs[1].values())
plt.hist([temp,temp2,temp3], color=['r','g','b'], alpha=0.5, normed=1)
plt.xlabel("Eccentricity")
plt.ylabel("Frequency")
first =mpatches.Patch(color="r", label="Hadza H Network")
second = mpatches.Patch(color="g", label=f"Average of 100 Randomized Hadza H Networks")
third = mpatches.Patch(color = 'b', label=f"Small-World Network")
plt.legend(loc='upper right', handles=[first, second,third])
plt.title("Histogram of the Eccentricity in the different networks")
plt.show()

'''
temp= list(clusts[0].values())
temp2=clustran
temp3=list(clusts[1].values())
plt.hist([temp,temp2,temp3], color=['r','g','b'], alpha=0.5, normed=1)
plt.xlabel("Clustering Coefficient")
plt.ylabel("Frequency")
first =mpatches.Patch(color="r", label="Hadza H Network")
second = mpatches.Patch(color="g", label=f"Average of 100 Randomized Hadza H Networks")
third = mpatches.Patch(color = 'b', label=f"Small-World Network")
plt.legend(loc='upper right', handles=[first, second,third])
plt.title("Histogram of the clustering coefficients of the nodes in the different networks")
plt.show()


#print(clusts)
print("Clustering coefficient")
for clust in clusts:
    temp=list(clust.values())
    print(statistics.mean(temp))
print(statistics.mean(clustran))
#print(degs)
temp= [d[1] for d in degs[0]]
temp2=degsran
temp3=[d[1] for d in degs[1]]
plt.hist([temp,temp2,temp3], color=['r','g','b'], alpha=0.5, normed=1)
plt.xlabel("Degrees")
plt.ylabel("Frequency")
first =mpatches.Patch(color="r", label="Hadza H Network")
second = mpatches.Patch(color="g", label=f"Average of 100 Randomized Hadza H Network")
third = mpatches.Patch(color = 'b', label=f"Small-World Network")
plt.legend(loc='upper right', handles=[first, second,third])
plt.title("Histogram of the degree distribution in the different networks")
plt.show()
quit()

#Network efficiency
#Gini Coefficient/Weight distribution
