import pandas
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import statistics

flatresults=[]
for j in range(20,21):
    csvFile = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/coopstats{j}.csv')
    col = len(csvFile.columns)
    csvFile = csvFile.iloc[:,100:col]
    x=csvFile.values.tolist()
    flatresults.append([item for sublist in x for item in sublist])

'''
#first=[item for sublist in flatresults[0:99] for item in sublist]
second = [item for sublist in flatresults[1:101] for item in sublist]

flatresults = [flatresults[0], second]
'''
sns.set(style="darkgrid")
cols=["blue", "green", "coral", "mediumpurple", "red", "mediumpurple", "aquamarine", "indigo", "palegreen", "yellow"]

lines = 'Statistical Results:'
with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/Stats.txt', 'w') as f:
    f.write(lines)

counter=0
for res in flatresults:
    if counter<1:
        col = "red"
    else:
        col="green"
    sns.histplot(data=res, x=res, kde=True, bins = 100, stat='density', label=f'Run: {counter}')

    mean=statistics.mean(res)
    var=statistics.stdev(res)
    print(mean)
    more_lines=[f"Run number {counter}", f"Mean = {mean}", f"Standard deviation = {var}"]
    with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/stats.txt', 'a') as f:
        for line in more_lines:
            f.write(line)
            f.write('\n')
    counter+=1
first =mpatches.Patch(color="red",label="Hadza Honey Network")
second = mpatches.Patch(color="green",label=f"Randomized Hadza Honey Networks")
plt.xlabel("Cooperation rate")
plt.ylabel("Density")
plt.xlim(0,1)
plt.legend(loc='upper right')#, handles=[first, second]) 
plt.title("Comparison of the Honey Hadza Network vs 100 Randomized Hadza Honey Networks")
plt.show()
