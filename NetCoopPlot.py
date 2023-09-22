import networkx as nx
from matplotlib import pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as lines
import pandas

cmap1=plt.cm.RdBu
#make plot to see cooperation of nodes in network
def NetPlot(G, means, gen):
    figfsm = plt.figure()  #prepare plot

    figfsm.set_figwidth(15)
    figfsm.set_figheight(10)
    
    nodecolors = {n: means[n] for n in range(len(means))}
    labels = {n: str(round(means[n],2)) for n in range(len(means))} #label all nodes with their respective cooperationratio
    nx.draw_networkx_nodes(G, pos, node_color = [nodecolors[node] for node in G.nodes()] , cmap = cmap1, vmin = 0, vmax = 0.75)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels) #add network graph to plot
    fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/netplot{e, gen}.png"
    plt.savefig(fname)  
    plt.clf()

#make plot to see change of cooperation of nodes in network over two different conditions
def CompareNodeCoop(G, means1, means2):
    mean = means2-means1
    print(means2)
    print(means1)
    print(mean)
    figfsm = plt.figure()  #prepare plot
    figfsm.set_figwidth(15)
    figfsm.set_figheight(10)
    #plot the average cooperativness of the nodes in the network
    labels = {n: str(round(mean[n+1],2)) for n in range(len(mean)-1)} #label all nodes with their respective cooperationratio
    nx.draw_networkx_nodes(G,pos,node_color=[mean[n+1] for n in range(nx.number_of_nodes(G))],cmap = cmap1,vmin=-0.5,vmax=0.5)
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G, pos,labels) #add network graph to plot
    plt.show()


net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/agtanet.txt'
G = nx.read_weighted_edgelist(net,nodetype = int) #read in network


net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/mappedAgta.txt'
G = nx.read_weighted_edgelist(net,nodetype = int) #read in network
pos = nx.spring_layout(G,seed=1)

net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaRanNet/AgtaRan{0}.txt'
#G = nx.read_weighted_edgelist(net,nodetype = int) #read in network
'''mapping = {}
nodelist=list(G.nodes)
for i in range(nx.number_of_nodes(G)):
    mapping[nodelist[i]]=i
G=nx.relabel_nodes(G, mapping) #relabel nodes '''


for e in range(4):
    nodescoop = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/netnodecoop{e}.csv')
    for l in range(15):
        means=(nodescoop.iloc[(l*10):(l+1)*10,1:].mean(axis=0))
        #print(means)
        NetPlot(G,means, l*10)
quit()
e=1
j=0
csvFile2 = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/NodeCoopstats{e, j}.csv')
means2=csvFile2.mean(axis=0)
print(means2)
e=0
csvFile1 = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/NodeCoopstats{e, j}.csv')
means1=csvFile1.mean(axis=0)
CompareNodeCoop(G,means1,means2)


