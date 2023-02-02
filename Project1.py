import random
import numpy as np
import networkx as nx
import statistics
from matplotlib import pyplot as plt
#import math
from copy import deepcopy
from  matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import pandas as pd
from itertools import combinations
#import cProfile

#Create colourmap for network graph
cmap1=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)

#creates a finite state machine (fsm)
class fsm:
    def __init__(self, dim):
        self.curstate = 0   #To keep track of which state fsm is in
        self.dim = dim      #How complex is the fsm
        self.points = 0     #Keep track of payoff

        keys = [strats[p] for p in chostrats]
        key = random.choice(keys)


        if key == 0:
            self.states = list(np.random.randint(0,2,dim)) #Randomly create the different states. Can be either 1 = cooperate or 0 = defect

            self.coop = list(np.random.randint(0,dim,dim))  # Randomly create the state changes if opponent cooperates

            self.defect = list(np.random.randint(0,dim,dim))#Randomly create the state changes if opponent defects
        
        elif key == 1:  #Fc
            self.states = [1]
            self.coop =[0]
            self.defect =[0]
        elif key ==2: #FD
            self.states = [0]
            self.coop =[0]
            self.defect =[0]
        elif key==3:  #TfT
            self.states=[1,0]
            self.coop = [0,0]
            self.defect = [1,1]
        elif key==4:  #WsLs
            self.states=[1,0]
            self.coop = [0,1]
            self.defect =[1,0]
        elif key==5:  #G
            self.states = [1,0]
            self.coop = [0,1]
            self.defect = [1,1]
        elif key == 6:           #Ex
            self.states=[0,1]
            self.coop = [0,1]
            self.defect = [1,0]
        elif key == 7:          #sus Coop
            self.states=[0,1]
            self.coop = [1,1]
            self.defect = [0,0]
        elif key == 8:          #sm Ex
            self.states = [0,1]
            self.coop = [0,0]
            self.defect = [0,1]

        #update state of fsm
    def update(self,coop):
        if coop:
            self.curstate = self.coop[self.curstate]
        else:
            self.curstate = self.defect[self.curstate]

        #mutate the fsm
    def mutate(self):
        if mutmode ==1:
            keys = [strats[p] for p in chostrats]
            key = random.choice(keys)
            if key ==1:
                self.states = [1]
                self.coop =[0]
                self.defect =[0]
            elif key ==2:
                self.states = [0]
                self.coop =[0]
                self.defect =[0]
            elif key==3:
                self.states=[1,0]
                self.coop = [0,0]
                self.defect = [1,1]
            elif key==4:  #WsLs
                self.states=[1,0]
                self.coop = [0,1]
                self.defect =[1,0]
            elif key==5:
                self.states = [1,0]
                self.coop = [0,1]
                self.defect = [1,1]
            elif key == 6:
                self.states=[0,1]
                self.coop = [0,1]
                self.defect = [1,0]
            elif key == 7:          #sus Coop
                self.states=[0,1]
                self.coop = [1,1]
                self.defect = [0,0]
            elif key == 8:          #sm Ex
                self.states = [0,1]
                self.coop = [0,0]
                self.defect = [0,1]
        elif mutmode==2:
            self.states = list(np.random.randint(0,2,self.dim)) #Randomly create the different states. Can be either 1 = cooperate or 0 = defect

            self.coop = list(np.random.randint(0,self.dim,self.dim))  # Randomly create the state changes if opponent cooperates

            self.defect = list(np.random.randint(0,self.dim,self.dim))#Randomly create the state changes if opponent defects
        else:

            for i in range(1): #Possibility of multiple mutations
                x = random.randint(0,3) #Randomly choose one of the four possible mutation types
                if x==0 and len(self.states)<topcomp:        #add one state
                    pos = random.randint(0,len(self.states)) #position of newly added state
                    temp = random.randint(0,len(self.states)-1) #choose random position to link to new state
                    self.states.insert(pos,random.randint(0,1)) #insert the new state
                    for i in range(len(self.coop)): #relink all the arrows to correct state
                        if self.coop[i] >= pos:
                            self.coop[i]+=1
                        if self.defect[i] >= pos:
                            self.defect[i]+=1
                    
                    if random.randint(0,1)==1:  #add link to new state in random position
                        self.coop[temp]=pos
                    else:
                        self.defect[temp]=pos
                    self.coop.insert(pos, random.randint(0,len(self.states)-1)) 
                    self.defect.insert(pos, random.randint(0,len(self.states)-1))

                #delete one state
                elif x==1 and len(self.states)>mincomp:
                    pos = random.randint(0,len(self.states)-1)
                    self.states.pop(pos)
                    self.coop.pop(pos)
                    self.defect.pop(pos)

                    for i in range(len(self.coop)): 
                        if self.coop[i] == pos:     
                            self.coop[i]=random.randint(0,len(self.states)-1) #if arrow to deleted state make random arrow
                        elif self.coop[i]>pos:
                            self.coop[i]-=1     #if arrow to state bigger than deleted make -1 so it still goes to same state
                        if self.defect[i] == pos:
                            self.defect[i]=random.randint(0,len(self.states)-1)
                        elif self.defect[i]>pos:
                            self.defect[i]-=1
                if self.curstate>len(self.states)-1:
                    self.curstate-=1

                #Redraw one random arrow
                elif x==2:
                    temp = random.randint(0,len(self.states)-1) #choose random position to redraw arrow
                    if random.randint(0,1)==1:  #change one arrow
                        self.coop[temp]=random.randint(0,len(self.states)-1)
                    else:
                        self.defect[temp]=random.randint(0,len(self.states)-1)

                #Change one random state
                elif x==3:
                    temp = random.randint(0,len(self.states)-1)
                    self.states[temp]=random.randint(0,1)

#calculates payoff of a round of PD
#input: tuple of 1 or 0
#output: tuple with payoffs for both players
def payoff(stratA,stratB):
    #1=cooperate
    #0=defect
    if stratA==stratB:
        if stratA==1:
            result=[povec[1],povec[1]]#both cooperate
        else:
            result=[povec[2],povec[2]]#both defect
    elif stratA==1:
        result=[povec[3],povec[0]]#A gets played
        
    else:
        result=[povec[0],povec[3]]#B gets played
    return (result)

#calculates payoff of a round of PD with kinselection
#input: tuple of 1 or 0
#output: tuple with payoffs for both players
def kinspayoff(stratA,stratB, r):
    #1=cooperate
    #0=defect
    if stratA==stratB:
        if stratA==1:
            result=[povec[1]+r*povec[1],povec[1]+r*povec[1]]#both cooperate
        else:
            result=[povec[2]+r*povec[2],povec[2]+r*povec[2]]#both defect
    elif stratA==1:
        result=[povec[3]+r*povec[0],povec[0]+r*povec[3]]#A gets played
        
    else:
        result=[povec[0]+r*povec[3],povec[3]+r*povec[0]]#B gets played
    return (result)

#kill players based on their payoff
#input: list with all the agents
#output: list with players and 0s for killed agents
def kill(players):

    count=0     #keep track how many get killed

    payoffs=[pl.points for pl in players]
    av=statistics.mean(payoffs)
    sd=statistics.stdev(payoffs)
    #for different Selection type
    #minpayoff=random.choices(payoffs, weights= payoffs)[0]#Randomly choosef
    #players[payoffs.index(minpayoff)] = 0

    for i in range(len(players)):
        if np.random.uniform(0,1)<selcoeff:
            if players[i].points<np.random.normal(av,sd+0.01) and count<len(players)-1:
                players[i] = 0
                count+=1
    #print(count)
    return(players)

#Moran Process
def moran(players):
    players[-1]=0
    return(players)

#Function that repopulates the player vector in a network
#Input: list with agents and 0s for killed agents, and network
#Output: list filled with agents
def NWrepopulate(players,G):
    for i in range(len(players)):
        if players[i] == 0:
            neighbour_list = []
            for neighbour in G[i]: # G[node] = all neighbours of node
                if players[neighbour] != 0:
                    for times in range(int(G[i][neighbour]['weight'])):
                        neighbour_list.append(neighbour) #create list of all possible parents, balanced by weight
            if neighbour_list == []:    #if no neighbours alive create random new agent
                players[i] = fsm(topcomp)
            else:
                parent = random.choice(neighbour_list) # then selects one neighbour from the list
                players[i] = deepcopy(players[parent])
                if random.uniform(0,1)<mutrate:
                    players[i].mutate()
            
        players[i].points = 0 #reset points to zero for next generation
    return players

#Function that visualizes a Finite State machine
#Input: finite state machine
#Output: graph of finite state machine
def visfsm(fsm):
    B = nx.DiGraph() #create diGraph to visualize fsm
    coopedge=[]
    defedge=[]
    for i in range(len(fsm.states)):
        B.add_node(i, pos=(i,0))    #Add state as node
        coopedge.append((i,fsm.coop[i]))    #Add cooperation edge
        defedge.append((i,fsm.defect[i]))   #Add defection edge

    pos=nx.get_node_attributes(B,"pos") #create position of network
    nx.draw_networkx_nodes(G,pos,
    nodelist=[i for i in range(len(fsm.states)) if fsm.states[i]==0],
    node_color="tab:red")   #draw red nodes for defection states
    nx.draw_networkx_nodes(G,pos,
    nodelist=[i for i in range(len(fsm.states)) if fsm.states[i]==1],
    node_color="tab:blue")  #draw blue nodes for cooperation states

    nx.draw_networkx_edges(B,pos, coopedge,edge_color="tab:blue",connectionstyle="arc3,rad=-0.4") #draw blue edges for cooperation edges
    nx.draw_networkx_edges(B,pos,defedge,edge_color="tab:red",connectionstyle="arc3,rad=0.5")   #draw red edges for defection edges
    return

#Function that checks if two fsm are the same:
#Input: tupel of two fsms
#Output: 1 if the same, 0 if not
def samefsm(fsmA, fsmB):
    if fsmA.states == fsmB.states and fsmA.coop == fsmB.coop and fsmA.defect == fsmB.defect:    #check if states, cooperation edges and defection edges are the same
        return 1
    else:
        return 0

#Function that tables the makeup of the population
#Input: list of agents
#Output: sorted nested list with example agent and instances in pop e.g. [[agentA, 5], [agentB, 3], ..., [agentZ, 1]]
def popwatch(players):
    playercomp=[[players[0],0]] #initialize list
    for player in players:
        for i in range(len(playercomp)):
            if samefsm(player, playercomp[i][0]): #if agent already in nested list add one to instances
                playercomp[i][1]+=1
                break
            elif i==len(playercomp)-1:
                playercomp.append([player,1])   #if agent not in nested list, add it to list
    
    playercomp.sort(reverse=True,key = lambda player: player[1]) #sort nested list by occurences
    return(playercomp)

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
 
#Function that simulates one round of PD
#Input: two agents
#Output: result for both players, updated state of player A
def PD(playerA,playerB,r):
    if kins:
        res = kinspayoff(playerA.states[playerA.curstate],playerB.states[playerB.curstate],r)
    else:
        res = payoff(playerA.states[playerA.curstate],playerB.states[playerB.curstate])
    Astate = playerA.curstate
    playerA.update(playerB.states[playerB.curstate])
    playerB.update(playerA.states[Astate])
    return(res,playerA.states[Astate])

#Function that lets two fsm play against each other
#Input: two agents, continuation probability
#Output: Payoffvector for A, Payoffvector for B, Cooperation vector for A, Count of how many rounds were played
def iteratedPD(playerA, playerB, alpha, r):

    #generate payoff vectors for both players
    payoffsA=[]
    payoffsB=[]
    coops=[]

    #reset current state of both players to zero
    playerA.curstate=0
    playerB.curstate=0

    rn=-1
    count=0
    while rn<alpha:
        #calculate payoff
        temp=PD(playerA,playerB,r)
        #append payoffs to the respective payoffvector
        payoffsA.append(temp[0][0])
        payoffsB.append(temp[0][1])
        coops.append(temp[1])
        rn=random.uniform(0,1)#chance of alpha to go into next round
        count+=1
        
    #calculate average payoff for both players
    resA=sum(payoffsA)/count
    resB=sum(payoffsB)/count
    return(payoffsA,payoffsB,coops,count)

#Function that simulates one generation of iterated PD in a Network
#Input: list with agents, amount of rounds to be played in gen, continuation probability, network 
#Output: list with updated agents
def genNetwPD(players,rounds,alpha,G):
    for pl in players:  #create cooperation and point lists for all agents
        pl.coops=[]
        pl.points=[]
    for r in range(rounds):
        random_node_list = random.sample(range(len(G)), len(G)) # select G nodes in random order
        for node in random_node_list: # for each node in random list
            neighbour_list = []
            temp=0
            neiweights=[n['weight'] for n in G[node].values()]#SLOW
            neighbours=list(G[node].keys())
            selected_neighbour = random.choices(neighbours,weights=neiweights)[0] # then selects one neighbour from the list
            #adjalpha = alpha**(10/int(G[node][selected_neighbour]['weight'])) #adjust alpha to scale with weight of connection
            r=0
            if kins:
                try:
                    r = R.edges[node,selected_neighbour]['weight'] 
                except:
                    r=0
            temp = iteratedPD(players[node],players[selected_neighbour],alpha, r)  #play iterated PD against selected neighbour
            players[node].points += temp[0]
            players[node].coops += temp[2]
        
    for pl in players:  #calculate average cooperationrate and average points for each player
        pl.coopratio= round(sum(pl.coops)/len(pl.coops),5)
        pl.points = round(sum(pl.points)/len(pl.points),5)
    return(players)        
        
#Function that simulates gens amount of ll of iterated PD over a network
#Input: list with agents, number of rounds per gen, number of generation, continuation prob, network
#Outpu: cool graphs/animations
def simNetwPD(players,rounds,gens,alpha,G):
    if animate:
        anfig=plt.figure()
    genscoops= []   #keep track of average cooperation rate per generation
    gencomps=[]
    nodescoop =np.zeros((gens,nx.number_of_nodes(G)))
    for k in range(gens): 
        players=genNetwPD(players,rounds,alpha,G) #let Agents play the generation

        #players.sort(reverse=True,key = lambda neuraln: neuraln.points) #sort agents by their payoff
        coopratios=[pl.coopratio for pl in players] #create list with all cooperation rates
        comps=[len(pl.states) for pl in players]
        genscoops.append(statistics.mean(coopratios))   #add average cooperation rate of generation to list
        gencomps.append(statistics.mean(comps))
        payoffs=[pl.points for pl in players]       #create list with all payoffs
        #print(statistics.mean(coopratios))
        #print(statistics.mean(comps))
        if popint:
            if k == 10:
                global gen10pop
                gen10pop=gen10pop+players
            if k == 20:
                global gen20pop
                gen20pop=gen20pop+players
            if k == 30:
                global gen30pop
                gen30pop=gen30pop+players
            if k == 40:
                global gen40pop
                gen40pop=gen40pop+players
            if k == 50:
                global gen50pop
                gen50pop=gen50pop+players
            if k == 100:
                global gen100pop
                gen100pop=gen100pop+players

        if animate:
            popwatched = popwatch(players)   #check makeup of population

            axgrid = anfig.add_gridspec(5,4) 
            labels = {n: str(round(players[n].coopratio,2)) for n in range(len(players))} #label all nodes with their respective cooperationratio
            anfig.add_subplot(axgrid[0:3,:])
            nx.draw_networkx_nodes(G,pos,node_color=[ag.coopratio for ag in players],cmap = cmap1,vmin=0,vmax=1)
            nx.draw_networkx_edges(G,pos)
            nx.draw_networkx_labels(G, pos,labels) #add network graph to plot

            anfig.add_subplot(axgrid[3:, :2])
            plt.plot(genscoops) #add cooperation over generations to plot
            plt.plot(gencomps)

            anfig.add_subplot(axgrid[3:, 2:])
            visfsm(popwatched[0][0]) #add most abundant agent to plot
            plt.pause(0.01)
            plt.clf()
        nodescoop[ (k,)]=np.asarray(coopratios)
        if ifmoran:
            players = moran(players)
        else:
            players = kill(players) #kill players
        players = NWrepopulate(players,G) #repopulate players
    #plt.show()
    avnodescoop=[statistics.mean(nodescoop[:,n]) for n in range(nx.number_of_nodes(G))]

    genscoops=[statistics.mean(nodescoop[gen,:]) for gen in range(gens)]

    
    return(nodescoop)#genscoops)

#Function that simulates sim amount of simNetwPD
#Input: Complexity, number of rounds per gen, continuation prob, number of generations, Network
#Output: cool graphs/animations
def xsims(comp,rounds, gens, alpha, G, sims):
    coopstats =np.zeros((sims,gens))
    nodecoopstats = np.zeros((sims,nx.number_of_nodes(G)))
    finalpop=[]

    #If there is interest in the population dynamics, keeps all the players at different times
    if popint:
        global gen10pop
        gen10pop=[]
        global gen20pop
        gen20pop=[]
        global gen30pop
        gen30pop=[]
        global gen40pop
        gen40pop=[]
        global gen50pop
        gen50pop=[]
        global gen100pop
        gen100pop=[]
    plt.clf() #prepare plot
    figfsm.set_figwidth(15)
    figfsm.set_figheight(10)

    #main loop that runs all the simulations
    for sim in range(sims):
        print(sim)
        agents=[]
        #create the agents for the simulation
        for j in range(nx.number_of_nodes(G)):
            agents.append(fsm(comp))       #Randomly create nodes
        nodescoop=simNetwPD(agents,rounds,gens,alpha,G)
        coopstats[sim,]=np.asarray([statistics.mean(nodescoop[gen,:]) for gen in range(gens)])
        nodecoopstats[sim,]=np.asarray([statistics.mean(nodescoop[:,n]) for n in range(nx.number_of_nodes(G))])
        plt.plot(coopstats[sim,:], color="lightsteelblue")
        avcoop=[]
        sdcoop=[]
        avnodecoop=[statistics.mean(nodecoopstats[:,n])for n in range(nx.number_of_nodes(G))]
        finalpop=finalpop+agents
    for gen in range(gens):
        avcoop.append(statistics.mean(coopstats[:,gen]))
        sdcoop.append(statistics.stdev(coopstats[:,gen]))
    avcoop=np.asarray(avcoop)
    sdcoop=np.asarray(sdcoop)
    
    #create final plot
    plt.plot(avcoop, linewidth=3, color="red")
    plt.plot(avcoop+sdcoop,linewidth=2, color="red", linestyle="dashed")
    plt.plot(avcoop-sdcoop,linewidth=2, color="red", linestyle="dashed")
    plt.xlabel("Generations")
    plt.ylim(0,1)
    plt.ylabel("Cooperation rate")
    fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/sim_{i}.png"
    plt.savefig(fname)  
    plt.clf()

    #create plot of final population
    figfsm.set_figwidth(4)
    figfsm.set_figheight(15)
    axgridfsm = figfsm.add_gridspec(5,1) 
    for z in range(5):
        try:
            figfsm.add_subplot(axgridfsm[z,:],title=f"{round(popwatch(finalpop)[z][1]/(sims*nx.number_of_nodes(G))*100,2)}%")
            visfsm(popwatch(finalpop)[z][0])
        except:
            print("meh")
    fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/sim_{i}_endpop.png"
    plt.savefig(fname)  
    plt.clf()

    #if interest in population dynamics, plot all the pops at different times
    if popint:
        figfsm.set_figwidth(4)
        figfsm.set_figheight(15)
        axgridfsm = figfsm.add_gridspec(5,1) 
        for z in range(5):
            try:
                figfsm.add_subplot(axgridfsm[z,:],title=f"{round(popwatch(gen10pop)[z][1]/(sims*nx.number_of_nodes(G))*100,2)}%")
                visfsm(popwatch(gen10pop)[z][0])
            except:
                print("meh")
        fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/sim_{i}_gen10pop.png"
        plt.savefig(fname)  
        plt.clf()

        figfsm.set_figwidth(4)
        figfsm.set_figheight(15)
        axgridfsm = figfsm.add_gridspec(5,1) 
        for z in range(5):
            try:
                figfsm.add_subplot(axgridfsm[z,:],title=f"{round(popwatch(gen20pop)[z][1]/(sims*nx.number_of_nodes(G))*100,2)}%")
                visfsm(popwatch(gen20pop)[z][0])
            except:
                print("meh")
        fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/sim_{i}_gen20pop.png"
        plt.savefig(fname)  
        plt.clf()

        figfsm.set_figwidth(4)
        figfsm.set_figheight(15)
        axgridfsm = figfsm.add_gridspec(5,1) 
        for z in range(5):
            try:
                figfsm.add_subplot(axgridfsm[z,:],title=f"{round(popwatch(gen30pop)[z][1]/(sims*nx.number_of_nodes(G))*100,2)}%")
                visfsm(popwatch(gen30pop)[z][0])
            except:
                print("meh")
        fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/sim_{i}_gen30pop.png"
        plt.savefig(fname)  
        plt.clf()

        figfsm.set_figwidth(4)
        figfsm.set_figheight(15)
        axgridfsm = figfsm.add_gridspec(5,1) 
        for z in range(5):
            try:
                figfsm.add_subplot(axgridfsm[z,:],title=f"{round(popwatch(gen40pop)[z][1]/(sims*nx.number_of_nodes(G))*100,2)}%")
                visfsm(popwatch(gen40pop)[z][0])
            except:
                print("meh")
        fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/sim_{i}_gen40pop.png"
        plt.savefig(fname)  
        plt.clf()

        figfsm.set_figwidth(4)
        figfsm.set_figheight(15)
        axgridfsm = figfsm.add_gridspec(5,1) 
        for z in range(5):
            try:
                figfsm.add_subplot(axgridfsm[z,:],title=f"{round(popwatch(gen50pop)[z][1]/(sims*nx.number_of_nodes(G))*100,2)}%")
                visfsm(popwatch(gen50pop)[z][0])
            except:
                print("meh")
        fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/sim_{i}_gen50pop.png"
        plt.savefig(fname)  
        plt.clf()


        figfsm.set_figwidth(4)
        figfsm.set_figheight(15)
        axgridfsm = figfsm.add_gridspec(5,1) 
        for z in range(5):
            try:
                figfsm.add_subplot(axgridfsm[z,:],title=f"{round(popwatch(gen100pop)[z][1]/(sims*nx.number_of_nodes(G))*100,2)}%")
                visfsm(popwatch(gen100pop)[z][0])
            except:
                print("meh")
        fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/sim_{i}_gen100pop.png"
        plt.savefig(fname)  
        plt.clf()

    figfsm.set_figwidth(15)
    figfsm.set_figheight(10)
    #plot the average cooperativness of the nodes related to their degrees
    numneighs = [G.degree[node] for node in range(len(G))]
    plt.scatter(numneighs,avnodecoop)
    plt.xlabel("Number of neighbours")
    plt.xlim(0,40)
    plt.ylim(0,1)
    plt.ylabel("Cooperation rate")
    fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/neighbourstocoop_{i}.png"
    plt.savefig(fname)  
    plt.clf()


    figfsm.set_figwidth(15)
    figfsm.set_figheight(10)
    #plot the average cooperativness of the nodes in the network
    labels = {n: str(round(avnodecoop[n],2)) for n in range(len(agents))} #label all nodes with their respective cooperationratio
    nx.draw_networkx_nodes(G,pos,node_color=[avnodecoop[n] for n in range(nx.number_of_nodes(G))],cmap = cmap1,vmin=0,vmax=1)
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G, pos,labels) #add network graph to plot
    fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/networksim_{i}.png"
    plt.savefig(fname)  
    plt.clf()

    #save coopstats:
    DF = pd.DataFrame(coopstats)
    DF.to_csv(F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/coopstats{i}.csv")
    return(avcoop)
    #plt.show()

animate = 0 #1 if we want animation, 0 if not
ifmoran = 0

net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/agtanet.txt'
G = nx.read_weighted_edgelist(net,nodetype = int) #read in network

Rnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/Redglist.txt'
R = nx.read_weighted_edgelist(Rnet, nodetype = int)

Hadzanet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/HoneyHadza.txt'
G = nx.read_weighted_edgelist(Hadzanet, nodetype=int)

HadzaRnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/HoneyRHadza.txt'
R = nx.read_weighted_edgelist(HadzaRnet, nodetype=int)
'''
largest_cc = max(nx.connected_components(G), key=len)
G=G.subgraph(largest_cc).copy()
nx.set_edge_attributes(G, values = 1, name = 'weight')

 
Hadzanet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/CampHadza.txt'
G = nx.read_edgelist(Hadzanet, nodetype=int)

HadzaRnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaNetwork/CampRHadza.txt'
R = nx.read_weighted_edgelist(HadzaRnet, nodetype=int)

largest_cc = min(nx.connected_components(G), key=len)
G=G.subgraph(largest_cc).copy()'''
#nx.set_edge_attributes(G, values = 1, name = 'weight')

  
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

#create comparable small world network
#G = nx.watts_strogatz_graph(Rsize, int(2*Redges/Rsize), 0)#int(2*edges/size)

#R=nx.gnm_random_graph(Rsize,Redges)
#create comparable random network

#G=nx.gnm_random_graph(size,edges)
#create comparable small world network
#G = nx.watts_strogatz_graph(size, int(2*edges/size), 0.1)#int(2*edges/size)
#nx.set_edge_attributes(G, values = 1, name = 'weight')

pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes


posstrats=["nFSM","AC","AD","TfT","WsLs","G","Ex","susC","smEx"]
strats=dict(nFSM=0,AC=1,AD=2,TfT=3,WsLs=4,G=5,Ex=6,susC=7,smEx=8)
'''combs=[]
for l in range(6):
    combs +=list(combinations(posstrats[3:9],l+1))
for q in range(len(combs)):
    combs[q]=list(combs[q])'''

figfsm = plt.figure()  #prepare plot
'''
numneighs = [G.degree[node] for node in range(len(G))]
plt.hist(numneighs)
plt.xlabel("Number of neighbours")
plt.xlim(0,40)
plt.ylim(0,20)
plt.show()
'''
lines = 'Simulation parameters:'
with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/siminfo.txt', 'w') as f:
    f.write(lines)
metacops=[]
for i in range(101):
    b = 5
    c = 1
    povec=(b,b-c,0,-c)
    selcoeff = 0.4
    mutrate = 0.5
    topcomp=3
    chostrats=[posstrats[0]]
    mutmode = 0
    rounds=1
    gens = 250
    alpha=0.9
    sims = 1000
    kins = 0
    popint=0
    mincomp=1

    #G = nx.watts_strogatz_graph(size, int(2*edges/size), 0.1)#int(2*edges/size)
    #nx.set_edge_attributes(G, values = 1, name = 'weight')
    #G = addWeights(weights,G)
    pos = nx.spring_layout(G, seed=3113794652)
    
    if i<1:
        pass    
    else:
        sims=10
        net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/HadzaHRanNet/HadzaHRan{i-1}.txt'
        G = nx.read_weighted_edgelist(net,nodetype = int) #read in network
        #G = nx.watts_strogatz_graph(size, int(2*edges/size), 0.02*(i))#int(edges/size)
        #G=nx.gnm_random_graph(size,edges)
        #nx.set_edge_attributes(G, values = 1, name = 'weight')
        pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
        #G = addWeights(weights,G)
        
        pass
    print(f"---------{i}----------")
    metacops.append(xsims(topcomp,rounds,gens,alpha,G,sims)) #start simulation
    #Write down simulation parameters
    more_lines=[F'Sim num = {i}', F'Rounds = {rounds}',
        F'Generations = {gens}', F'Alpha = {alpha}', F'Simulations = {sims}', 
        F'Strategies = {chostrats}', F'Benefit to cost ration = {b/c}',
        F'Selection coefficient = {selcoeff}', F'Complexity = {topcomp}',
        F'Mutation rate = {mutrate}', F'Mutation mode = {mutmode}', F'Kinselection = {kins}', '-----------------------']
    with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/siminfo.txt', 'a') as f:
        for line in more_lines:
            f.write(line)
            f.write('\n')


    #G = nx.watts_strogatz_graph(size, int(2*edges/size), 0.1)#int(edges/size)
    #G=nx.gnm_random_graph(size,edges)
    #nx.set_edge_attributes(G, values = 1, name = 'weight')
    #G=shuffleWeights(G)
    


count=0

for rate in metacops:
    if count<1: 
        col="red"
        chostrats=[posstrats[0]]
    else:
        col="green"
    plt.plot(rate, color=col,linewidth=2)#, label=f'SAPL: {Sapls[count]}')
    first =mpatches.Patch(color="red",label=f"Hadza Honey Network")
    second = mpatches.Patch(color="green",label=f"Randomized Hadza Honey Networks")
    plt.ylim(0,1)
    plt.ylabel("Cooperation rate")
    plt.xlabel("Generation")
    plt.legend(loc='upper right', handles=[first, second])
    plt.title("Hadza Honey network to randomized Hadza Honey network")
    count+=1
plt.plot(metacops[0],linewidth=2,color="red")
plt.savefig("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/final.png")

