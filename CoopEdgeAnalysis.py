import pandas
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import statistics
import networkx as nx
import numpy as np



for e in range(2):
    for j in range(1):
        edgecoop = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/IRAgta/Edgecoopstats{e, j}.csv')

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
        avcoop.to_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/IRAgta/edgecoops{e, j}.csv')
quit()