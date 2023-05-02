import networkx as nx
from matplotlib import pyplot as plt

G=nx.read_weighted_edgelist(f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/IndRecNetAnalysis/NetworkSim0Gen{0}.txt", nodetype=int)
pos = nx.spring_layout(G, seed=3113794652)
plt.show()
for i in range(100):
    G = nx.read_weighted_edgelist(f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/IndRecNetAnalysis/NetworkSim0Gen{100*i}.txt", nodetype=int)
    widths = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_nodes(G,pos,node_color="red")
    nx.draw_networkx_edges(G,pos,edgelist = widths.keys(), width=list(widths.values()), alpha=0.3)
    nx.draw_networkx_labels(G,pos)
    plt.pause(0.1)
    plt.clf()
                                  