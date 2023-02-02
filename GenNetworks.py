import networkx as nx
import random
from matplotlib import pyplot as plt

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
        rec=list(neighbours)
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
            
            '''if len(list(nx.connected_components(F)))>1:
                print([node,neighneigh],[neigh, nodeneigh])
                print(neighbours)
                print(rec)
                pos = nx.spring_layout(F, seed=3113794652)
                nx.draw_networkx_nodes(F,pos,node_color="red")
                nx.draw_networkx_edges(F,pos)
                nx.draw_networkx_labels(F,pos)
                plt.show()
                print([node,neighneigh],[neigh, nodeneigh])'''
                
    
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

#Function that randomizes the relationship network of a social network
#Input: Social network, Relationship network
#Output: Randomized Relationship network
def Kinrandomizer(G,R):
    pass

net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/agtanet.txt'
G = nx.read_weighted_edgelist(net,nodetype = int) #read in network

'''
Hadzanet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/CampHadza.txt'
G = nx.read_edgelist(Hadzanet, nodetype=int)

Hadzanet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/HoneyHadza.txt'
G = nx.read_weighted_edgelist(Hadzanet, nodetype=int)



#largest_cc = min(nx.connected_components(G), key=len)
#G=G.subgraph(largest_cc).copy()

'''


#nx.set_edge_attributes(G, values = 1, name = 'weight')

mapping = {}
nodelist=list(G.nodes)
for i in range(nx.number_of_nodes(G)):
    mapping[nodelist[i]]=i
G=nx.relabel_nodes(G, mapping) #relabel nodes 

pos = nx.spring_layout(G, seed=3113794652)
nx.draw_networkx_nodes(G,pos,node_color="red")
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos)
plt.show()
print(len(G.edges()))
for i in range(0,100):
    F=randomizer(G)
    #nx.set_edge_attributes(F, values = 1, name = 'weight')
    nx.write_weighted_edgelist(F, f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/AgtaRanNet/AgtaRan{i}.txt")
    nx.draw_networkx_nodes(F,pos,node_color="red")
    nx.draw_networkx_edges(F,pos)
    nx.draw_networkx_labels(F,pos)
    #plt.show()
    print(i)


