import pandas
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

start=100
flatresults=[]
for j in range(5):
    csvFile = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/coopstats{j}.csv')
    col = len(csvFile.columns)
    csvFile = csvFile.iloc[:,100:col]
    x=csvFile.values.tolist()
    flatresults.append([item for sublist in x for item in sublist])


sns.set(style="darkgrid")
cols=["blue", "green", "coral", "mediumpurple", "red", "mediumpurple", "aquamarine", "indigo", "palegreen", "yellow"]

counter=0
for res in flatresults:
    if counter<1:
        col = "forestgreen"
    else:
        col="orange"
    sns.histplot(data=res, x=res, color=cols[counter], kde=True, bins = 100, stat='density')
    counter+=1
first =mpatches.Patch(color="forestgreen",label="Weighted Hadza network with original Kin selection")
second = mpatches.Patch(color="orange",label=f"Weighted Hadza network with randomized Kin selection")
plt.xlabel("Cooperation rate")
plt.ylabel("Density")
plt.xlim(0,1)
#plt.legend(loc='upper right', handles=[first, second]) 
plt.show()
