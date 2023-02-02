import networkx as nx
import random
from matplotlib import pyplot as plt


probs = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7,1]

count=0
networks = []
for prob in probs:
    G = nx.watts_strogatz_graph(1000, 10, prob)
    networks.append(G)
    nx.write_weighted_edgelist(G, f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/SMWnets/smw{count}.txt")
    count+=1
