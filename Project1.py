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
import math

#creates a finite state machine (fsm)
class fsm:
    def __init__(self, dim):
        self.curstate = 0   #To keep track of which state fsm is in
        self.dim = dim      #How complex is the fsm
        self.points = 0     #Keep track of payoff

        #If only certain strategies are allowed
        keys = [strats[p] for p in chostrats]
        key = random.choice(keys)

        #Generates a second FSM/strategy for related agents if Kin Selection is activated
        if kins:
            self.KSstates = list(np.random.randint(0,2,dim)) #Randomly create the different states. Can be either 1 = cooperate or 0 = defect

            self.KScoop = list(np.random.randint(0,dim,dim))  # Randomly create the state changes if opponent cooperates

            self.KSdefect = list(np.random.randint(0,dim,dim))#Randomly create the state changes if opponent defects
        
        #Generates another FSM/strategy for more cooperative agents if Indirect Reciprocity is activated
        if irs:
            self.IRstates = list(np.random.randint(0,2,dim)) #Randomly create the different states. Can be either 1 = cooperate or 0 = defect

            self.IRcoop = list(np.random.randint(0,dim,dim))  # Randomly create the state changes if opponent cooperates

            self.IRdefect = list(np.random.randint(0,dim,dim))#Randomly create the state changes if opponent defects

        #Key = 0 means unrestricted strategy space so just generate a FSM
        if key == 0:
            self.states = list(np.random.randint(0,2,dim)) #Randomly create the different states. Can be either 1 = cooperate or 0 = defect

            self.coop = list(np.random.randint(0,dim,dim))  # Randomly create the state changes if opponent cooperates

            self.defect = list(np.random.randint(0,dim,dim))#Randomly create the state changes if opponent defects
        

        elif key == 1:  #generate a Full Cooperator FSM
            self.states = [1]
            self.coop =[0]
            self.defect =[0]
        elif key ==2: #generate a Full Defector FSM
            self.states = [0]
            self.coop =[0]
            self.defect =[0]
        elif key==3:  #Generate a Tit for Tat FSM
            self.states=[1,0]
            self.coop = [0,0]
            self.defect = [1,1]
        elif key==4:  #Generate a Win stay Lose shift FSM
            self.states=[1,0]
            self.coop = [0,1]
            self.defect =[1,0]
        elif key==5:  #Generate a Grim FSM
            self.states = [1,0]
            self.coop = [0,1]
            self.defect = [1,1]
        elif key == 6:  #Generate Exploit FSM
            self.states=[0,1]
            self.coop = [0,1]
            self.defect = [1,0]
        elif key == 7:   #Generate suspicious cooperator FSM
            self.states=[0,1]
            self.coop = [1,1]
            self.defect = [0,0]
        elif key == 8:   #Generate smart exploiter
            self.states = [0,1]
            self.coop = [0,0]
            self.defect = [0,1]


    #update state of the fsm after a normal round of prisoners dilemma
    def update(self,coop):
        if coop:
            self.curstate = self.coop[self.curstate]
        else:
            self.curstate = self.defect[self.curstate]
    #update state of the KS fsm after a round of prisoners dilemma against a relative
    def KSupdate(self, coop):
        if coop:
            self.curstate = self.KScoop[self.curstate]
        else:
            self.curstate = self.KSdefect[self.curstate]
    #update state of the IR fsm after a round of prisoners dilemma against a nice opponent
    def IRupdate(self, coop):
        if coop:
            self.curstate = self.IRcoop[self.curstate]
        else:
            self.curstate = self.IRdefect[self.curstate]

    
    #mutate the fsm during reproduction
    def mutate(self):

        #Mutmode = 1 means only certain strats (chostrats) are allowed
        if mutmode ==1:
            #generate new random strategy
            keys = [strats[p] for p in chostrats]
            key = random.choice(keys)
            
            #generate that FSM
            if key ==1: #FC
                self.states = [1]
                self.coop =[0]
                self.defect =[0]
            elif key ==2:   #FD
                self.states = [0]
                self.coop =[0]
                self.defect =[0]
            elif key==3:    #TfT
                self.states=[1,0]
                self.coop = [0,0]
                self.defect = [1,1]
            elif key==4:  #WsLs
                self.states=[1,0]
                self.coop = [0,1]
                self.defect =[1,0]
            elif key==5:    #Grim
                self.states = [1,0]
                self.coop = [0,1]
                self.defect = [1,1]
            elif key == 6:  #Exploit
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
        
        #Mutmode=2 means that a new random FSM is generated and replaces the old one
        elif mutmode==2:
            self.states = list(np.random.randint(0,2,self.dim)) #Randomly create the different states. Can be either 1 = cooperate or 0 = defect

            self.coop = list(np.random.randint(0,self.dim,self.dim))  # Randomly create the state changes if opponent cooperates

            self.defect = list(np.random.randint(0,self.dim,self.dim))#Randomly create the state changes if opponent defects
            if irs == 1:
                self.IRstates = list(np.random.randint(0,2,self.dim)) #Randomly create the different states. Can be either 1 = cooperate or 0 = defect

                self.IRcoop = list(np.random.randint(0,self.dim,self.dim))  # Randomly create the state changes if opponent cooperates

                self.IRdefect = list(np.random.randint(0,self.dim,self.dim))#Randomly create the state changes if opponent defects
            if irs == 1:
                self.KSstates = list(np.random.randint(0,2,self.dim)) #Randomly create the different states. Can be either 1 = cooperate or 0 = defect

                self.KScoop = list(np.random.randint(0,self.dim,self.dim))  # Randomly create the state changes if opponent cooperates

                self.KSdefect = list(np.random.randint(0,self.dim,self.dim))#Randomly create the state changes if opponent defects
        
        #Mutmode = 0 is a slight mutation from the original FSM
        else:
            for i in range(1): #Possibility of multiple mutations
                x = random.randint(0,3) #Randomly choose one of the four possible mutation types
                 #add one random state
                if x==0 and len(self.states)<topcomp:
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
                    pos = random.randint(0,len(self.states)-1) #select random state to delete
                    #delete state and links from this state
                    self.states.pop(pos)
                    self.coop.pop(pos)
                    self.defect.pop(pos)
                    #relink arrows to correct states
                    for i in range(len(self.coop)): 
                        if self.coop[i] == pos:     
                            self.coop[i]=random.randint(0,len(self.states)-1) #if arrow to deleted state make random arrow
                        elif self.coop[i]>pos:
                            self.coop[i]-=1     #if arrow to state bigger than deleted make -1 so it still goes to same state
                        if self.defect[i] == pos:
                            self.defect[i]=random.randint(0,len(self.states)-1)
                        elif self.defect[i]>pos:
                            self.defect[i]-=1
                #update curstate to correct position
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

            #Do the same mutation procedure for the KS FSM
            if kins:
                for i in range(1): #Possibility of multiple mutations
                    x = random.randint(0,3) #Randomly choose one of the four possible mutation types
                    if x==0 and len(self.KSstates)<topcomp:        #add one state
                        pos = random.randint(0,len(self.KSstates)) #position of newly added state
                        temp = random.randint(0,len(self.KSstates)-1) #choose random position to link to new state
                        self.KSstates.insert(pos,random.randint(0,1)) #insert the new state
                        for i in range(len(self.KScoop)): #relink all the arrows to correct state
                            if self.KScoop[i] >= pos:
                                self.KScoop[i]+=1
                            if self.KSdefect[i] >= pos:
                                self.KSdefect[i]+=1
                        
                        if random.randint(0,1)==1:  #add link to new state in random position
                            self.KScoop[temp]=pos
                        else:
                            self.KSdefect[temp]=pos
                        self.KScoop.insert(pos, random.randint(0,len(self.KSstates)-1)) 
                        self.KSdefect.insert(pos, random.randint(0,len(self.KSstates)-1))

                    #delete one state
                    elif x==1 and len(self.KSstates)>mincomp:
                        pos = random.randint(0,len(self.KSstates)-1)
                        self.KSstates.pop(pos)
                        self.KScoop.pop(pos)
                        self.KSdefect.pop(pos)

                        for i in range(len(self.KScoop)): 
                            if self.KScoop[i] == pos:     
                                self.KScoop[i]=random.randint(0,len(self.KSstates)-1) #if arrow to deleted state make random arrow
                            elif self.KScoop[i]>pos:
                                self.KScoop[i]-=1     #if arrow to state bigger than deleted make -1 so it still goes to same state
                            if self.KSdefect[i] == pos:
                                self.KSdefect[i]=random.randint(0,len(self.KSstates)-1)
                            elif self.KSdefect[i]>pos:
                                self.KSdefect[i]-=1
                    if self.curstate>len(self.KSstates)-1:
                        self.curstate-=1

                    #Redraw one random arrow
                    elif x==2:
                        temp = random.randint(0,len(self.KSstates)-1) #choose random position to redraw arrow
                        if random.randint(0,1)==1:  #change one arrow
                            self.KScoop[temp]=random.randint(0,len(self.KSstates)-1)
                        else:
                            self.KSdefect[temp]=random.randint(0,len(self.KSstates)-1)

                    #Change one random state
                    elif x==3:
                        temp = random.randint(0,len(self.KSstates)-1)
                        self.KSstates[temp]=random.randint(0,1)
            #Do the same Mutation procedure for the IR FSM
            if irs:
                for i in range(1): #Possibility of multiple mutations
                    x = random.randint(0,3) #Randomly choose one of the four possible mutation types
                    if x==0 and len(self.IRstates)<topcomp:        #add one state
                        pos = random.randint(0,len(self.IRstates)) #position of newly added state
                        temp = random.randint(0,len(self.IRstates)-1) #choose random position to link to new state
                        self.IRstates.insert(pos,random.randint(0,1)) #insert the new state
                        for i in range(len(self.IRcoop)): #relink all the arrows to correct state
                            if self.IRcoop[i] >= pos:
                                self.IRcoop[i]+=1
                            if self.IRdefect[i] >= pos:
                                self.IRdefect[i]+=1
                        
                        if random.randint(0,1)==1:  #add link to new state in random position
                            self.IRcoop[temp]=pos
                        else:
                            self.IRdefect[temp]=pos
                        self.IRcoop.insert(pos, random.randint(0,len(self.IRstates)-1)) 
                        self.IRdefect.insert(pos, random.randint(0,len(self.IRstates)-1))

                    #delete one state
                    elif x==1 and len(self.IRstates)>mincomp:
                        pos = random.randint(0,len(self.IRstates)-1)
                        self.IRstates.pop(pos)
                        self.IRcoop.pop(pos)
                        self.IRdefect.pop(pos)

                        for i in range(len(self.IRcoop)): 
                            if self.IRcoop[i] == pos:     
                                self.IRcoop[i]=random.randint(0,len(self.IRstates)-1) #if arrow to deleted state make random arrow
                            elif self.IRcoop[i]>pos:
                                self.IRcoop[i]-=1     #if arrow to state bigger than deleted make -1 so it still goes to same state
                            if self.IRdefect[i] == pos:
                                self.IRdefect[i]=random.randint(0,len(self.IRstates)-1)
                            elif self.IRdefect[i]>pos:
                                self.IRdefect[i]-=1
                    if self.curstate>len(self.IRstates)-1:
                        self.curstate-=1

                    #Redraw one random arrow
                    elif x==2:
                        temp = random.randint(0,len(self.IRstates)-1) #choose random position to redraw arrow
                        if random.randint(0,1)==1:  #change one arrow
                            self.IRcoop[temp]=random.randint(0,len(self.IRstates)-1)
                        else:
                            self.IRdefect[temp]=random.randint(0,len(self.IRstates)-1)

                    #Change one random state
                    elif x==3:
                        temp = random.randint(0,len(self.IRstates)-1)
                        self.IRstates[temp]=random.randint(0,1)

#calculates payoff of a round of PD
#input: tuple of 1 or 0
#output: tuple with payoffs for both players
def payoff(stratA,stratB):
    #1=cooperate
    #0=defectc
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
    payoffs=[pl.points for pl in players]   #get all the payoffs of the players
    enumerated_list = list(enumerate(payoffs))
    sorted_list = sorted(enumerated_list, key=lambda x: (x[1], random.random()))
    '''#MoranProcess
    nppayoffs=np.array(payoffs)
    minpos=np.array(np.where(nppayoffs == nppayoffs.min()))[0]
    killspot = np.random.choice(minpos,size=1)[0]
    players[killspot] = 0
    '''#Generate mean and standard deviation of the payoffs
    av=statistics.mean(payoffs)
    sd=statistics.stdev(payoffs)
    #Go through all the players, randomly decide if agent is subject to selection, then randomly decide if agent dies
    number = min(len(players)-1,np.random.normal(kills,kills/5))
    while count<number:
        random_number = random.randint(0,len(players)-1)
        if payoffs[random_number]<np.random.normal(av,sd+0.01) and players[random_number]!=0:
            players[random_number]=0
            count+=1
            '''
    for fr in range(len(sorted_list)):
        if np.random.uniform(0,1)<selcoeff: #selcoeff guide how strong selection
            #if sorted_list[fr][1]<np.random.normal(av,sd+0.01) and count<len(players)-1:    #check if points of player are smaller than a random number generated from normal dis with mean and sd from payoffs
            if count<kills and count<len(players)-1:
                players[sorted_list[fr][0]] = 0 #set agent to zero, deleted
                count+=1'''
    #print(count)
    return(players)

#Function that repopulates the player vector in a network
#Input: list with agents and 0s for killed agents, and network
#Output: list filled with agents
def NWrepopulate(players,G):
    for i in range(len(players)):   #go through all the positions in the player vector
        if players[i] == 0: #see if agent was deleted
            neighbour_list = []
            for neighbour in G[i]: # get all the neighbours of the node
                if players[neighbour] != 0: #check that neighbour isn't also deleted
                    for times in range(int(G[i][neighbour]['weight'])):
                        neighbour_list.append(neighbour) #create list of all possible parents, balanced by weight
            if neighbour_list == []:    #if no neighbours alive create random new agent
                non_zero_list = [x for x in players if x != 0]
                parent = random.choice(non_zero_list) # then selects one neighbour from the list based on weights
                players[i] = deepcopy(parent)  #copy parent into offsprings position
                if random.uniform(0,1)<mutrate: #mutate offspring based on mutation rate
                    players[i].mutate()
            else:
                parent = random.choice(neighbour_list) # then selects one neighbour from the list based on weights
                players[i] = deepcopy(players[parent])  #copy parent into offsprings position
                if random.uniform(0,1)<mutrate: #mutate offspring based on mutation rate
                    players[i].mutate()
            
        players[i].points = 0 #reset points to zero for  all the players for the next generation
    return players

#Function that makes a plot of the network with the cooperation rate of the nodes
#Input: Network and list of the cooperative values
#Output: plot of coop rate in network
def NetPlot(G, means, gen):
    figfsm = plt.figure()  #prepare plot

    figfsm.set_figwidth(15)
    figfsm.set_figheight(10)
    #plot the average cooperativness of the nodes in the network
    labels = {n: str(round(means[n],2)) for n in range(len(means))} #label all nodes with their respective cooperationratio
    nx.draw_networkx_nodes(G,pos,node_color=[means[n] for n in range(nx.number_of_nodes(G))],cmap = cmap1,vmin=0,vmax=1)
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G, pos,labels) #add network graph to plot
    fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/netplot{e,i,gen}.png"
    plt.savefig(fname)  
    plt.clf()

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
#Output: sorted nested list with example agent and number of instances in pop e.g. [[agentA, 5], [agentB, 3], ..., [agentZ, 1]]
def popwatch(players):
    playercomp=[[players[0],0]] #initialize list with first agent
    for player in players:  #go through all the players
        for i in range(len(playercomp)):  #check if agent already in nested list  
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
    weights=getWeights(G)   #get weights of network
    random.shuffle(weights) #shuffle weights

    addWeights(weights,G)   #add weights to network
    return(G)

#Funtion that gets the Weights of a network
#Input: Network
#Output: list with all the weights
def getWeights(R):
    weights=nx.get_edge_attributes(R,'weight')  #get weights of network
    weights=list(weights.values())      #turn into list
    random.shuffle(weights)     #shuffle weights
    return(weights)

#Function that adds weights to a network
#Input: Weights and network
#Output: Network with weights
def addWeights(weights,G):
    mapping = {}
    edgelist=list(G.edges)
    mapping=dict(zip(edgelist,weights)) #create dictionary with edges as keys and weights as values
    nx.set_edge_attributes(G, values = mapping, name = 'weight')    #add weights to network using dictionary
    return(G)
 
#Function that simulates one round of PD
#Input: two agents
#Output: result for both players, updated state of player A
def PD(playerA,playerB,r):
    #One round if Kin selection is on and players are related
    if kins and r>0:
        res = kinspayoff(playerA.KSstates[playerA.curstate], playerB.KSstates[playerB.curstate],r)  #get payoff in KS
        Astate = playerA.curstate   #Get state of player A
        Bstate= playerB.curstate    #Get state of player B
        playerA.KSupdate(playerB.KSstates[playerB.curstate])    #Update KSstate of player A
        playerB.KSupdate(playerA.KSstates[Astate])  #Update KSstate of player B
        newAstate = playerA.KSstates[Astate]    #save if A cooperated (1 or 0)
        newBstate = playerB.KSstates[Bstate]    #save if B cooperated (1 or 0)

    #One round if Indirect Reciprocity is activated
    elif irs and hasattr(playerB, "coopratio"): #Check if opponent has coopratio (doesn't have coopratio in first generation)
        if playerB.coopratio>avgencoops:    #If coopratio of opponent in last generation was greater than the average use IR FSM
            res = payoff(playerA.IRstates[playerA.curstate], playerB.IRstates[playerB.curstate]) #get payoff of Prisoner's Dilemma
            Astate = playerA.curstate   #Get state of player A
            Bstate= playerB.curstate    #Get state of player B
            playerA.IRupdate(playerB.IRstates[playerB.curstate]) #Update IRstate of player A
            playerB.IRupdate(playerA.IRstates[Astate])  #Update IRstate of player B
            newAstate = playerA.IRstates[Astate]    #save if A cooperated (1 or 0)
            newBstate = playerB.IRstates[Bstate]    #save if B cooperated (1 or 0)
        else:   #If coopratio of opponent in last generations was smaller than the average use normal FSM
            res = payoff(playerA.states[playerA.curstate],playerB.states[playerB.curstate]) #get payoff of Prisoner's Dilemma
            Astate = playerA.curstate   #Get state of player A
            Bstate= playerB.curstate    #Get state of player B
            playerA.update(playerB.states[playerB.curstate])    #Update state of player A
            playerB.update(playerA.states[Astate])    #Update state of player B
            newAstate = playerA.states[Astate]    #save if A cooperated (1 or 0)
            newBstate = playerB.states[Bstate]    #save if B cooperated (1 or 0)
    #One round if everything is normal
    else:
        res = payoff(playerA.states[playerA.curstate],playerB.states[playerB.curstate]) #get payoff of Prisoner's Dilemma
        Astate = playerA.curstate    #Get state of player A
        Bstate= playerB.curstate    #Get state of player B
        playerA.update(playerB.states[playerB.curstate])    #Update state of player A
        playerB.update(playerA.states[Astate])    #Update state of player B
        newAstate = playerA.states[Astate]    #save if A cooperated (1 or 0)
        newBstate = playerB.states[Bstate]    #save if B cooperated (1 or 0)
    return(res,newAstate, newBstate)    #Return result of prisoners dilemma and if a and b cooperated or not

#Function that lets two fsm play against each other
#Input: two agents, continuation probability, relatedness
#Output: Payoffvector for A, Payoffvector for B, Cooperation vector for A, Count of how many rounds were played
def iteratedPD(playerA, playerB, alpha, r):

    #generate payoff vectors for both players
    payoffsA=[]
    payoffsB=[]
    #generate cooperation lists for both players
    coopsA=[]
    coopsB=[]

    #reset current state of both players to zero
    playerA.curstate=0
    playerB.curstate=0

    rn=-1#initialize
    count=0
    while rn<alpha: 
        #calculate payoff
        temp=PD(playerA,playerB,r)
        #append payoffs to the respective payoffvector
        payoffsA.append(temp[0][0])
        payoffsB.append(temp[0][1])
        #append cooperation to respective lists
        coopsA.append(temp[1])
        coopsB.append(temp[2])
        rn=random.uniform(0,1)#chance of alpha to go into next round
        count+=1

    return(payoffsA,payoffsB,coopsA,count,coopsB)   #return pyofflists, cooperation lists and number of rounds

#Function that simulates one generation of iterated PD in a Network
#Input: list with agents, amount of rounds to be played in gen, continuation probability, network 
#Output: list with updated agents
def genNetwPD(players,rounds,alpha,G):
    edgecoop = pd.DataFrame(0, index=G.edges(),columns=["coops", "plays","coopav"])#generate dataframe to store edgecooperation values
    #rownames are tuple of nodes connected by edge, then one column for cooperation and one for number of rounds
    for pl in players:  #create cooperation and point lists for all agents
        pl.coops=[]
        pl.points=[]
    for r in range(rounds): #go through desired number of rounds
        random_node_list = random.sample(range(len(G)), len(G)) # select G nodes in random order
        for node in random_node_list: # for each node in random list let them play
            neighbour_list = []
            temp=0
            neiweights=[n['weight'] for n in G[node].values()]#generate weight list for neighbours
            neighbours=list(G[node].keys()) #get neighbours
            selected_neighbour = random.choices(neighbours,weights=neiweights)[0] # then selects one random neighbour from the list based on weight

            if kins:
                try:
                    r = R.edges[node,selected_neighbour]['weight'] #if KS is on and r > 0 get r from Kinship network    
                except:
                    r=0 #else r is equal to zero
            temp = iteratedPD(players[node],players[selected_neighbour],alpha, r)  #play iterated PD against selected neighbour
            players[node].points += temp[0] #append points to player
            players[node].coops += temp[2]  #append coops to player
            cooptemp = sum(temp[2]) + sum(temp[4])  #calculate number of cooperation moves over edge during this round
            #attach this number to the correct position in edgecoop, and number of rounds
            try:
                edgecoop.at[(node, selected_neighbour), "coops"]+=cooptemp
                edgecoop.at[(node, selected_neighbour), "plays"]+=temp[3]
            except:
                edgecoop.at[(selected_neighbour,node), "coops"]+=cooptemp
                edgecoop.at[(selected_neighbour,node), "plays"]+=temp[3]
    edgecoop.at[:,"coopav"]=edgecoop["coops"]/(2*edgecoop["plays"]) #average cooperation by dividing coops by plays after all the rounds
    for pl in players:  #calculate average cooperationrate and average points for each player
        pl.coopratio= round(sum(pl.coops)/len(pl.coops),5)
        pl.points = round(sum(pl.points)/len(pl.points),5)

    return(players, edgecoop["coopav"])         #return player list and the average cooperation for every edge vector  
        
#Function that simulates gens amount of generations of iterated PD over a network
#Input: list with agents, number of rounds per gen, number of generation, continuation prob, network
#Outpu: cool graphs/animations
def simNetwPD(players,rounds,gens,alpha,orG):
    G=orG.copy()    #copy network
    if animate: #prepare figure if animation wanted
        anfig=plt.figure()
    genscoops= []   #keep track of average cooperation rate per generation
    gencomps=[]     #keep track of average complexity of the fsm per generation
    nodescoop =np.zeros((gens,nx.number_of_nodes(G)))   #data frame to keep track of cooperation rate of nodes, gen rows, nodes as coloumns
    edgescoop = pd.DataFrame(0, index=G.edges(),columns=np.arange(gens))    #data frame to keep track of cooperation rate over edge, edges as rows and gens as coloumns
    global avgencoops
    avgencoops = 0  #generate global average cooperation rate
    global sdgencoops 
    sdgencoops= 0#generate global standard deviation cooperation rate

    for k in range(gens):   #go through all the generations
        temp = genNetwPD(players,rounds,alpha,G) #let Agents play the generation
        players= temp[0]
        edgescoop[k]=temp[1]
        coopratios=[pl.coopratio for pl in players] #create list with all cooperation rates
        comps=[len(pl.states) for pl in players]    #create list with all the complexities
        avgencoops = statistics.mean(coopratios)
        sdgencoops = statistics.stdev(coopratios)
        genscoops.append(avgencoops)   #add average cooperation rate of generation to list
        gencomps.append(statistics.mean(comps)) #add average complexity of generation
        if popint:  #if interest in strategies, save populations at different time spots
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

        if animate: #if animation on 
            popwatched = popwatch(players)   #check makeup of population

            axgrid = anfig.add_gridspec(5,4)    #make grid
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
        nodescoop[(k,)]=np.asarray(coopratios)
        players = kill(players) #kill players
        players = NWrepopulate(players,G) #repopulate players
    '''if netdev:
        for l in range(15):
            means=(nodescoop[(l*10):(l+1)*10, :].mean(axis=0))
            NetPlot(G,means, l*10)'''
    return(nodescoop, edgescoop)

#Function that simulates sim amount of simNetwPD
#Input: Complexity, number of rounds per gen, continuation prob, number of generations, Network
#Output: cool graphs/animations
def xsims(comp,rounds, gens, alpha, G, sims):
    coopstats =np.zeros((sims,gens))    #create dataframe to store average cooperation per gen for each simulation
    nodecoopstats = np.zeros((sims,nx.number_of_nodes(G)))  #create dataframe to store average cooperation of each node per simulation
    edgecoopstats = pd.DataFrame(0, index=G.edges(),columns=np.arange(sims))    #create dataframe to store average cooperation over each edge per simulation
    netnode =[]
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
            agents.append(fsm(comp))       #Randomly create agents with desired complexity
        temp=simNetwPD(agents,rounds,gens,alpha,G)  #run one simulation
        #save data
        nodescoop=temp[0]
        edgescoop=temp[1]
        coopstats[sim,]=np.asarray([statistics.mean(nodescoop[gen,:]) for gen in range(gens)])
        nodecoopstats[sim,]=np.asarray([statistics.mean(nodescoop[:,n]) for n in range(nx.number_of_nodes(G))])
        if netdev:
            netnode.append(nodescoop)
        #Calculate the average cooperation rate of each edge over the simulation
        for edge in range(nx.number_of_edges(G)):
            
            temp=[n for n in edgescoop.iloc[edge] if n==n]
            if len(temp)>0:
                edgecoopstats.iloc[edge,sim]=statistics.mean(temp)

        plt.plot(coopstats[sim,:], color="lightsteelblue")
        avcoop=[]  
        sdcoop=[]
        avnodecoop=[statistics.mean(nodecoopstats[:,n])for n in range(nx.number_of_nodes(G))]

    for gen in range(gens):
        avcoop.append(statistics.mean(coopstats[:,gen]))
        if len(coopstats[:,gen])==1:
            sdcoop.append(0)
        else:
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
    fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/sim_{e,i}.png"
    plt.savefig(fname)  
    plt.clf()

    if netdev:
        netnodecoop = np.zeros((gens,nx.number_of_nodes(G)))  #create dataframe to store average cooperation of each node per generation
        for g in range(gens):
            temp = np.zeros((sims, nx.number_of_nodes(G)))  #matrix to store cooperation for each node in all the sims in one generation
            for s in range(sims):
                temp[s,:] = netnode[s][g,:]
                genmeans = temp.mean(axis=0)
            netnodecoop[g,:]=genmeans   
        #save coopstats:
        DF = pd.DataFrame(netnodecoop)
        DF.to_csv(F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/netnodecoop{e}.csv")
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
    '''
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
    plt.clf()'''

    '''
    figfsm.set_figwidth(15)
    figfsm.set_figheight(10)
    #plot the average cooperativness of the nodes in the network
    labels = {n: str(round(avnodecoop[n],2)) for n in range(len(agents))} #label all nodes with their respective cooperationratio
    nx.draw_networkx_nodes(G,pos,node_color=[avnodecoop[n] for n in range(nx.number_of_nodes(G))],cmap = cmap1,vmin=0,vmax=1)
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G, pos,labels) #add network graph to plot
    fname = F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/networksim_{e,i}.png"
    plt.savefig(fname)  
    plt.clf()
    '''
    #save coopstats:
    DF = pd.DataFrame(coopstats)
    DF.to_csv(F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/coopstats{e,i}.csv")
    #save edgecoopstats:
    edgecoopstats.to_csv(F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/Edgecoopstats{e,i}.csv")
    nodecoopstatspd = pd.DataFrame(nodecoopstats)
    nodecoopstatspd.to_csv(F"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/NodeCoopstats{e,i}.csv")
    return(avcoop)
    #plt.show()

animate = 0 #1 if we want animation, 0 if not

net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/mappedAgta.txt'
G = nx.read_weighted_edgelist(net,nodetype = int) #read in network

Rnet = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/mappedAgtaKin.txt'
R = nx.read_weighted_edgelist(Rnet, nodetype = int)

pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
posstrats=["nFSM","AC","AD","TfT","WsLs","G","Ex","susC","smEx"]    #all the possible hard coded strats
strats=dict(nFSM=0,AC=1,AD=2,TfT=3,WsLs=4,G=5,Ex=6,susC=7,smEx=8)   #keys for all the possible strats

'''combs=[]
for l in range(6):
    combs +=list(combinations(posstrats[3:9],l+1))
for q in range(len(combs)):
    combs[q]=list(combs[q])'''

figfsm = plt.figure()  #prepare plot

#Create colourmap for network graph
cmap1=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)

rules = ["Net","SMW","IR","DR","KS"] #all the possible rules in the simulation
rules = ["Net","IR","DR","KS"]
#create all the different rule combinations
combs=[[]]
for l in range(5):
    combs +=list(combinations(rules,l+1))
for q in range(len(combs)):
    combs[q]=list(combs[q])
dels=[]
for comb in range(len(combs)):
    if "Net" in combs[comb] and "SMW" in combs[comb]:
        dels.append(comb)

for i in sorted(dels, reverse=True):
    del combs[i]
print(combs)

alphas = [0.99, 0.97, 0.95, 0.93, 0.9, 0.7, 0.5, 0.3, 0] #different alphas
b = 0.8     #benefit of cooperation in prisoner's dilemma
c = 0.2     #cost of cooperation in prisoner's dilemma
povec=(b,b-c,0,-c)  #payoffvector of prisoner's dilemma c(DC, CC, DD, CD)
selcoeff = 0.4      #selection coefficient if 0 no selection, if 1 on average half of pop is killed
mutrate = 0.5       #chance for mutation during reproduction
topcomp=3           #top complexity of fsm
mincomp=1           #minimum complexity of fsm
chostrats=[posstrats[0]]    #chosen strats for simulation posstrats=["nFSM","AC","AD","TfT","WsLs","G","Ex","susC","smEx"]  
mutmode = 0       #chosen mutation mode for simulation 0 = small mutation from original, 1 = mutation to other hardcoded strats, 2 = mutation to new random generated strat
rounds=1            #rounds per generation  
gens = 250     #generations per simulation
alpha=0             #continuation probability
sims = 10        #how many simulations
kins = 0            #if 1 KS is activated 0 not
popint=0            #interest in which strategies evolve         
chosrules = []      #chosen rules for this simulation
netdev = 0

#write down simulation parameters for each simulation
lines = 'Simulation parameters:'
with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/siminfo.txt', 'w') as f:
    f.write(lines)
metacops=[] #keep track of mean cooperation over different simulations

for e in range(20,21):    #cycle through different meta simulations
    '''chosrules = combs[e]
    sims=1
    nets = 100
    if "Net" in chosrules:
        sims=100
        nets = 1'''
    sims=1
    nets=100
    chosrules = []
    kills= int(len(G.nodes())*(e*0.05))
    selcoeff=1

    for i in range(37,nets):          #cycle through simulations in different networks (100 for ran/smw, 1 for agta)
        #reset different rules to standard
        Rnet = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaRanR/AgtaRanR{i}.txt'
        R = nx.read_weighted_edgelist(Rnet, nodetype = int)
        net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaRanNet/AgtaRan{i}.txt'
        G = nx.read_weighted_edgelist(net,nodetype = int) #read in network
        irs = 0
        alpha = 0
        kins = 0
        #G = nx.watts_strogatz_graph(size, int(2*edges/size), 0.1)#int(2*edges/size)
        #G = addWeights(weights,G)
        #pos = nx.spring_layout(G, seed=3113794652)

        #activate chosen rules
        for rule in chosrules:
            if rule == 'Net': #load Agta networks
                net = 'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/mappedAgta.txt'
                G = nx.read_weighted_edgelist(net,nodetype = int) #read in network
                Rnet = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/mappedAgtaKin.txt'
                R = nx.read_weighted_edgelist(Rnet, nodetype = int)
                '''net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/SMWnets/smw{e}.txt'
                G = nx.read_weighted_edgelist(net,nodetype = int) #read in network
                nx.set_edge_attributes(G, values = 1, name = 'weight')
                Rnet = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/SMWnets/smwR{e}.txt'
                R = nx.read_weighted_edgelist(Rnet,nodetype = int) #read in network'''


            if rule == "SMW":   #load smw networks
                Rnet = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaSMW/AgtaWRSMW{i}.txt'
                R = nx.read_weighted_edgelist(Rnet, nodetype = int)
                net = f'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/networks/AgtaSMW/AgtaSMW{i}.txt'
                G = nx.read_weighted_edgelist(net,nodetype = int) #read in network
            elif rule == 'IR':  #activate Indirect Reciprocity
                irs = 1
            elif rule == 'DR':  #activate direct reciprocity
                alpha = 0.9
            elif rule == 'KS':  #activate Kin Selection
                kins = 1
        print(f"---------{e,i}----------")
        metacops.append(xsims(topcomp,rounds,gens,alpha,G,sims)) #run simulation
        #Write down simulation parameters
        more_lines=[F'Sim num = {e,i}', F'Rounds = {rounds}',
            F'Generations = {gens}', F'Alpha = {alpha}', F'Simulations = {sims}', 
            F'Strategies = {chostrats}', F'Benefit to cost ration = {b/c}',
            F'Selection coefficient = {selcoeff}',F'Complexity = {topcomp}',
            F'Mutation rate = {mutrate}', F'Mutation mode = {mutmode}', F'Kinselection = {kins}', 
            F'Indirect Reciprocity = {irs}', F'Rules = {chosrules}', '-----------------------']
        with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/siminfo.txt', 'a') as f:
            for line in more_lines:
                f.write(line)
                f.write('\n')
    
count=0

#make final plot
for rate in metacops:

    plt.plot(rate,linewidth=2)#,label=f'Alpha = {alphas[count]}')
    names=[]
    names.append(mpatches.Patch(color="red",label=f"Kin Selection in Randomized Agta Networks"))
    names.append(mpatches.Patch(color="blue",label=f"Kin Selection in the Agta Network"))
    names.append(mpatches.Patch(color="green", label =f"Multilevel Agta Network with Kin Selection"))
    plt.ylim(0,1)
    plt.ylabel("Cooperation rate")
    plt.xlabel("Generation")
    plt.legend(loc='upper right')#, handles= names)
    plt.title("Impact of Direct Reciprocity in the Agta Network")
    count+=1
#plt.plot(metacops[0],linewidth=2,color="red")
plt.savefig("C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/final.png")
