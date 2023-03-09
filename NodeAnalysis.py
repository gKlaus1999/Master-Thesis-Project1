import pandas
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import statistics
import networkx as nx
import numpy as np

net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/agtanet.txt'
G = nx.read_weighted_edgelist(net,nodetype = int) #read in network

net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/Redglist.txt'
R = nx.read_weighted_edgelist(net,nodetype = int) #read in network

mapping = {}
pos = nx.spring_layout(G, seed=3113794652)
nodelist=list(G.nodes)
for i in range(nx.number_of_nodes(G)):
    mapping[nodelist[i]]=i
G=nx.relabel_nodes(G, mapping) #relabel nodes 
R=nx.relabel_nodes(R,mapping)

degs=list(G.degree())
deg=[d[1] for d in degs]
clust=(list(nx.clustering(G, weight='weight').values()))
ecc=(list(nx.eccentricity(G).values()))


for j in range(13):
    nodecoop = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/NodeCoopstats{j}.csv')

    names = list(nodecoop.columns)
    names[0]="sim"
    names[1:len(names)]=[int(n)for n in names[1:len(names)]]
    nodecoop.columns=names
    avcoop=pandas.DataFrame()
    avcoop["nodes"]=names[1:len(names)]

    temp = []
    for node in range(len(nodecoop.axes[1])-1):
        #print(nodecoop.loc[np.arange(len(nodecoop.axes[0])),node])
        temp.append(statistics.mean(nodecoop.loc[np.arange(len(nodecoop.axes[0])),node]))
    avcoop["coop"]=temp
    avcoop["deg"]=deg
    avcoop["clust"]=clust
    avcoop["ecc"]=ecc
    print(avcoop)

    avcoop.to_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/Results/NodeAnalysis/nodecoops{j}.csv')
quit()