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
G=nx.relabel_nodes(G, mapping) #relabel no  des 
R=nx.relabel_nodes(R,mapping)
print(mapping)

#nx.write_weighted_edgelist(G, f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/edgeAnalysis/mappedAgta.txt")


j=1
for j in range(3):
    edgecoop = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/Edgecoopstats{j}.csv')

    names = list(edgecoop.columns)
    names[0]="v"
    names[1]="u"
    names[2:len(names)] = [int(number) for number in names[2:len(names)]]
    edgecoop.columns=names
    avcoop=edgecoop[["u","v"]]
    avcoop["coop"]=np.nan
    temp = []
    for edge in range(len(edgecoop.axes[0])):
        temp.append(statistics.mean(edgecoop.loc[edge, np.arange(len(edgecoop.axes[1])-2)]))
    avcoop["coop"]=temp
    print(avcoop)
    avcoop.to_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/edgecoops{j}.csv')
quit()