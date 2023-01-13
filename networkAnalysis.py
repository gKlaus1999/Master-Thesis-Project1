import networkx as nx
import random
import statistics

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

#Function that randomizes a network
#Input: Network
#Output: randomized network
def randomizer(G):
    for node in range(G.number_of_nodes()):
        neighbours=list(G[node].keys())
        for neigh in range(len(neighbours)):
            pass


net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/agtanet.txt'
G = nx.read_weighted_edgelist(net,nodetype = int) #read in network

Rnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/Redglist.txt'
R = nx.read_weighted_edgelist(Rnet, nodetype = int)
'''
Hadzanet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/HadzaNet.txt'
G = nx.read_edgelist(Hadzanet, nodetype=int)

HadzaRnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/HadzaRnet.txt'
R = nx.read_weighted_edgelist(HadzaRnet, nodetype=int)

largest_cc = max(nx.connected_components(G), key=len)
G=G.subgraph(largest_cc).copy()
nx.set_edge_attributes(G, values = 1, name = 'weight')
'''
mapping = {}
pos = nx.spring_layout(G, seed=3113794652)
nodelist=list(G.nodes)
for i in range(nx.number_of_nodes(G)):
    mapping[nodelist[i]]=i
G=nx.relabel_nodes(G, mapping) #relabel nodes 
R=nx.relabel_nodes(R, mapping)

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

#BA = nx.barabasi_albert_graph(size,round(edges/size))
Ran=nx.gnm_random_graph(size,edges)
Ran=addWeights(weights,Ran)

#create comparable small world network
Smw = nx.watts_strogatz_graph(size, int(2*edges/size), 0.1)#int(2*edges/size)
Smw = addWeights(weights,Smw)

#create scale free graph
Sf = nx.scale_free_graph(size)
Sf = addWeights(weights,Sf)

G=addInvWeights(G)
networks = [G, Ran, Smw]
#Degree Distribution
totweights=list(G.degree(weight='weight'))

degs=[]
dens=[]
trans=[]
clusts=[]
eccs=[]
diams=[]
spls=[]
spls2=[]

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
    eccs.append(nx.eccentricity(network))

    #diameter
    diams.append(eccs[len(eccs)-1])

    #Shortest path length
    spls.append(statistics.mean(eccs[len(eccs)-1]))
    spls2.append(nx.average_shortest_path_length(G))#weight='invweight'

print(degs)

#Network efficiency
#Gini Coefficient/Weight distribution


