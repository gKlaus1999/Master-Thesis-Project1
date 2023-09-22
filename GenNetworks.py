import networkx as nx
import random
from matplotlib import pyplot as plt
import statistics
import pandas as pd
import numpy as np
from collections import Counter

#Function that checks if two nodes are connected:
#Input: touple of nodes
#Output: True or False
def nodes_connected(G, u, v):
    return u in G.neighbors(v)

#Function that divides a number n into k random parts that sum up to n
#Input: n and k
#Output: list of k numbers
def rearange(n,k):
    res = [1]*k
    for i in range(int((n-k))):
        r=random.randint(0,k-1)
        res[r] += 1
    return(res)

#Function that is one step in creating a comparable lattice network for a given network
#Input: Network G
#Output: Network G that is more lattice like than the original one
def onesteplatticeinator(G):
    for node in range(G.number_of_nodes()):
        for secnode in range(G.number_of_nodes()):
            if secnode==node:
                break
            neighbours=list(G[node].keys())
            neighbour = random.choice(neighbours)

            secneighbours=list(G[secnode].keys())
            secneighbour = random.choice(secneighbours)
            if neighbour==secnode or secneighbour==node or neighbour==secneighbour:
                break
            if not(nodes_connected(G,node,secnode)) and not(nodes_connected(G, neighbour,secneighbour)) and abs(node-secnode)+abs(neighbour-secneighbour) <= abs(node-neighbour)+abs(secnode-secneighbour):
                try:
                    G.remove_edge(node,neighbour)
                    G.remove_edge(secnode,secneighbour)
                    G.add_edge(node, secnode)
                    G.add_edge(neighbour, secneighbour)
                except:
                    print(node,neighbour, secnode, secneighbour)
    return(G)  

#Function that creates a comparable lattice network for a given network
#Input: Network G
#Output: Network F that is a comparable lattice network to original G
def latticeinator(G):
    tries=0
    while(tries<10):
        orclus = nx.average_clustering(G)
        newG = G.copy()
        newG = onesteplatticeinator(newG)
        newclus = nx.average_clustering(newG)
        if newclus > orclus:
            tries=0
            G = newG
            print(newclus)
        if newclus < orclus:
            tries+=1
            print(tries)
    return(G)

#Function that randomizes a network
#Input: Network
#Output: randomized network
def randomizer(b):
    F=b.copy()
    beacons = list(F.degree(weight="weight"))
    totalweights = [beacon[1] for beacon in beacons]
    print(sum(totalweights))
    for node in range(F.number_of_nodes()):
        neighbours=(F[node].keys())
        for neigh in range(len(neighbours)):
            check=[n for n in neighbours if n == neigh]
            if len(check)==1:
                    
                m=0
                switch=0
                while m <100 and not(switch):
                    nodeneighs = [n for n in neighbours if n != neigh]
                    if len(nodeneighs)==0:
                        break
                    nodeneigh = random.choice(nodeneighs)

                    neighneighs = F[nodeneigh].keys()
                    neighneighs = [n for n in neighneighs if n != node]
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
            else:
                pass
                  
    edgeslist=list(F.edges)
    weighsdic = dict.fromkeys(edgeslist,0)
    for beacon in beacons:
        edges = F.edges(beacon[0])
        ws = rearange(int(beacon[1]),len(edges))
        #print(ws)
        count=0
        for edge in edges:
            try:
                weighsdic[edge] += ws[count]/2
            except:
                weighsdic[edge[::-1]] += ws[count]/2
            count+=1
    for key in weighsdic:
        weighsdic[key]={"weight":weighsdic[key]}
    
    nx.set_edge_attributes(F,weighsdic)
    beacons = F.degree(weight="weight")
    totalweights = [beacon[1] for beacon in beacons]
    print(sum(totalweights))
    return(F)

#Function that randomizes the Weights of a network
#Input: Network
#Output: Network with randomized weights
def weightRan(b):
    F=b.copy()
    beacons = list(F.degree(weight="weight"))
    edgeslist=list(F.edges)
    weighsdic = dict.fromkeys(edgeslist,0)
    for beacon in beacons:
        edges = F.edges(beacon[0])
        ws = rearange(int(beacon[1]),len(edges))
        #print(ws)
        count=0
        for edge in edges:
            try:
                weighsdic[edge] += ws[count]/2
            except:
                weighsdic[edge[::-1]] += ws[count]/2
            count+=1
    for key in weighsdic:
        weighsdic[key]={"weight":weighsdic[key]}
    
    nx.set_edge_attributes(F,weighsdic)
    beacons = F.degree(weight="weight")
    totalweights = [beacon[1] for beacon in beacons]
    return(F)

#Funtion that gets the Weights of a network
#Input: Network
#Output: list with all the weights
def getWeights(R):
    weights=nx.get_edge_attributes(R,'weight')
    weights=list(weights.values())
    random.shuffle(weights)
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

#Function that randomizes the relationship network of a social network
#Input: Social network, Relationship network
#Output: Randomized Relationship network
def Kinrandomizer(G,R):
    weights=getWeights(R)
    edges=G.edges()
    Redges=random.sample(edges, len(weights))
    newR=nx.Graph()
    for edge in Redges:
        newR.add_edge(edge[0],edge[1])
    newR = addWeights(weights,newR)
    return(newR)

#Function that randomizes a relationship network one of a social network one step at a time
#Input: Social network, Kinship network
#Output: Randomized Kinship network
def onestepRan(G,R):
    #delete random weight edge pair
    edges = nx.get_edge_attributes(R, 'weight') 
    delEdge, w = random.choice(list(edges.items()))
    
    R.remove_edge(delEdge[0],delEdge[1])
    #add weight to another random edge from G that is not already in R
    Gedges = G.edges()
    newEdge = random.sample(R.edges(),1)

    while R.has_edge(newEdge[0][0], newEdge[0][1]):
        newEdge = random.sample(Gedges,1)
    R.add_edge(newEdge[0][0],newEdge[0][1], weight = w)
    return(R)

def stepCorDown(G,R):
    gdf = nx.to_pandas_edgelist(G)
    rdf = nx.to_pandas_edgelist(R)
    gdf = reorderR(gdf)
    rdf = reorderR(rdf)
    #gdf = reorderR(gdf)
    
    merged_df = pd.merge(rdf,gdf, on = ['source','target'], how='left')#different join so its only R>0
    merged_df =merged_df.fillna(0)
    merged_df = merged_df.sort_values(by=['weight_x', 'weight_y'], ascending=[False,True])#maybe change so high R at the top
    # Compute the weights using a probability distribution
    num_rows = len(merged_df)
    weights = np.arange(1, num_rows + 1)  # Assign higher weights to rows at the top
    
    # Normalize the weights to create a probability distribution
    weights = weights / np.sum(weights)
    weights = weights[::-1]
    # Randomly select a row based on the computed probability distribution
    thisR=0
    while thisR==0:
        random_row = np.random.choice(merged_df.index, p=weights)
        thisR = list(merged_df.loc[random_row,['weight_x']])
    selected_row = list(merged_df.loc[random_row, ['source','target']])
    R.remove_edge(selected_row[0],selected_row[1])

    '''mergnew = pd.merge(rdf,gdf, on = ['source','target'], how='right')#different join so its only R>0
    mergnew =mergnew.fillna(0)
    mergnew = mergnew.sort_values(by=['weight_x', 'weight_y'], ascending=[True,False])#maybe change so high R at the top
    
    # Compute the weights using a probability distribution
    num_rows = len(mergnew)
    wNew = np.arange(1, num_rows + 1)  # Assign higher weights to rows at the top
    
    # Normalize the weights to create a probability distribution
    wNew = wNew / np.sum(wNew)
    wNew = wNew[::-1]

    random_row = np.random.choice(mergnew.index, p=wNew)
    selected_row = list(mergnew.loc[random_row, ['source','target']])
    while R.has_edge(selected_row[0],selected_row[1]):
        random_row = np.random.choice(mergnew.index, p=wNew)
        selected_row = list(mergnew.loc[random_row, ['source','target']])
    R.add_edge(int(selected_row[0]), int(selected_row[1]), weight = thisR[0])'''
    
    Gedges = G.edges()
    newEdge = random.sample(R.edges(),1)

    while R.has_edge(newEdge[0][0], newEdge[0][1]):
        newEdge = random.sample(Gedges,1)
    R.add_edge(newEdge[0][0],newEdge[0][1], weight = thisR[0])
    return(R)

#Function that checks the Correlation between R and F
#Input: Social network, Kinship network
#Output: correlation between R and F
def checkCor(G,R):
    gdf = nx.to_pandas_edgelist(G)
    rdf = nx.to_pandas_edgelist(R)
    gdf = reorderR(gdf)
    rdf = reorderR(rdf)
    #gdf = reorderR(gdf)
    
    merged_df = pd.merge(rdf,gdf, on = ['source','target'], how='outer')
    merged_df =merged_df.fillna(0)
    merged_df = merged_df.sort_values(by=['weight_y','weight_x'])
    corr = merged_df['weight_x'].corr(merged_df['weight_y'])
    return(corr)

#Function that reorders the edgelist of a network¨
#Input: pd dataframe of the edgelist
#Output: sorted dataframe
def reorderR(rdf):
    for i in range(len(rdf.index)):
        if rdf.at[i,'source']>rdf.at[i,'target']:
            temp = rdf.at[i,'source']
            rdf.at[i,'source'] = rdf.at[i,'target']
            rdf.at[i,'target'] = temp
    rdf = rdf.sort_values(by=['source','target'])
    return(rdf)

#Function that generates a kinship network with correlation c
#Input: Social network, Kinship network, wished correlation
#Output: kinship network with desired correlation
def RanKS(G,f,c):
    R = f.copy()
    curc = checkCor(G,R)
    print(curc)

    while(curc<c):
        Rnew = onestepRan(G,R)
        newc = checkCor(G,R)
        if newc>=curc:
            R = Rnew
            curc=newc
        print(curc)
    return(R,curc)

#Function that generates a kinship network much smaller than original
#Input: Social network, Kinship network, desired correlation
#Output: Kinship network with desired correlation
def CorDown(G,f,c):

    R=f.copy()
    curc = checkCor(G,R)

    while(curc>c):
        Rnew = stepCorDown(G,R)
        newc = checkCor(G,R)
        if newc<=curc:
            R=Rnew
            curc=newc
        print(curc)

#Function that produces a KinNetwork for a SMW network comparable to AgtaKin
#Input: SMWnet, Agta KinNetwork
#Output: SMW kinnetwork
def SMWKin(S,R):
    weights=getWeights(R)
    countWeights = dict(Counter(weights))
    el=np.array(list(countWeights.keys()))
    counts=list(countWeights.values())
    r = [b*10.48218 for b in counts]
    r = [int(b) for b in r]

    newWeights= np.repeat(el, r, axis=0)

    edges=S.edges()
    Redges=random.sample(edges, len(newWeights))
    newR=nx.Graph()
    for edge in Redges:
        newR.add_edge(edge[0],edge[1])
    newR = addWeights(newWeights,newR)
    return(newR)

net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/agtanet.txt'
G = nx.read_weighted_edgelist(net,nodetype = int) #read in network

mapping = {}
nodelist=list(G.nodes)
for i in range(nx.number_of_nodes(G)):
    mapping[nodelist[i]]=i
G=nx.relabel_nodes(G, mapping) #relabel nodes 
pos = nx.spring_layout(G,seed=3113794652)

net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/mappedAgta.txt'
G = nx.read_weighted_edgelist(net, nodetype = int) #read in network

Rnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/mappedAgtaKin.txt'
R = nx.read_weighted_edgelist(Rnet, nodetype = int)
print(checkCor(G,R))

for i in range(21):
    print(i)
    net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/SMWnets/smw{i}.txt'
    S = nx.read_weighted_edgelist(net, nodetype = int) #read in network
    print(nx.average_shortest_path_length(S))
    print("------------")
#print(nx.sigma(G))

'''cors=[]
for i in range(100):
    net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaRanNet/AgtaRan{i}.txt'
    G = nx.read_weighted_edgelist(net, nodetype = int) #read in network

    Rnet = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaRanR/AgtaRanR{i}.txt'
    R = nx.read_weighted_edgelist(Rnet, nodetype = int)
    cors.append(checkCor(G,R))
print(cors)
print(statistics.mean(cors))
print(statistics.stdev(cors))
plt.hist(cors)
plt.show()
for i in range(21):
    print(i)
    net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/SMWnets/smw{i}.txt'
    S = nx.read_weighted_edgelist(net, nodetype = int) #read in network
    print(nx.sigma(S))
    print("------------")
#print(nx.sigma(G))

for i in range(21):
    net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/SMWnets/smw{i}.txt'
    S = nx.read_weighted_edgelist(net, nodetype = int) #read in network
    newR = SMWKin(S,R)
    nx.write_weighted_edgelist(newR,f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/SMWnets/smwR{i}.txt")

lines = 'Correlation Coefficients'
#with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/DifRcorr/Corrs.txt', 'w') as f:
 #   f.write(lines)
  #  f.write('\n')
for i in range(8):
    for j in range(3):
        lines = []
        print(f"---------{(i,j)}---------")
        p = RanKS(G,R, 0.2 + 0.05*i)
        k=p[0]
        cor = p[1]
        #nx.write_weighted_edgelist(k,f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/DifRcorr/DifRcorr{i,j}.txt")
        lines.append(f"network: {i,j}")
        lines.append(f"{cor}")
        #with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/DifRcorr/Corrs.txt', 'a') as f:
         #   for line in lines:
          #      f.write(line)
            #    f.write('\n')
quit()

weights = getWeights(G)
size = nx.number_of_nodes(G)
edges = nx.number_of_edges(G)
for i in range(100):
    F=nx.watts_strogatz_graph(size, int(2*edges/size), 0.1)
    F=addWeights(weights,F)

    B = Kinrandomizer(F,R)
    nx.write_weighted_edgelist(B,f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaSMW/AgtaWRSMW{i}.txt")
    nx.write_weighted_edgelist(F,f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaSMW/AgtaWSMW{i}.txt")



quit()
for i in range(2):
    R=nx.read_weighted_edgelist(f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/AgtaRanR/AgtaRanR{i}.txt", nodetype=int)
    R=nx.relabel_nodes(R,mapping)
    nx.write_weighted_edgelist(R,f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/edgeAnalysis/AgtaRanRmapped{i}.txt")

for i in range(100):
    newR=Kinrandomizer(G,R)
    pos = nx.spring_layout(newR, seed=3113794652)
    nx.write_weighted_edgelist(newR, f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/AgtaRanR/AgtaRanR{i}.txt")

mapping = {}
nodelist=list(G.nodes)
for i in range(nx.number_of_nodes(G)):
    mapping[nodelist[i]]=i
G=nx.relabel_nodes(G, mapping) #relabel nodes 

G=nx.gnm_random_graph(50,400)
pos = nx.spring_layout(G, seed=3113794652)
nx.draw_networkx_nodes(G,pos,node_color="red")
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()

G=latticeinator(G)
nx.draw_networkx_nodes(G,pos,node_color="red")
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()
print(nx.average_clustering(G))
quit()
for i in range(0,100):
    F=randomizer(G)
    #nx.set_edge_attributes(F, values = 1, name = 'weight')
    nx.write_weighted_edgelist(F, f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/AgtaRanNet/AgtaRan{i}.txt")
    nx.draw_networkx_nodes(F,pos,node_color="red")
    nx.draw_networkx_edges(F,pos)
    nx.draw_networkx_labels(F,pos)
    #plt.show()
    print(i)'''


