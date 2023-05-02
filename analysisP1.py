import pandas
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import statistics
from itertools import combinations
from scipy.stats import ttest_ind

#Produces a Plot of means and Standard deviation of cooperation over different alphas
def AlphaPlot(alphas, means, stds):
    sns.reset_defaults()
    alphas=[1-n for n in alphas]
    basemean=means[0]
    basestd=stds[0]
    #means =[m/basemean for m in means]
    #stds = [s/basestd for s in stds]
    plt.xscale("log")
    plt.xlabel("1 - Alpha")
    plt.ylabel("Cooperation Rate", color="blue")
    plt.tick_params(axis="y", labelcolor="blue")
    plt.plot(alphas,means,color="blue", linestyle = "dashed")

    plt.twinx()
    plt.ylabel("Standard Deviation", color = "red")
    plt.tick_params(axis="y", labelcolor="red")
    plt.plot(alphas,stds, linestyle="dashed", color="red")


    first =mpatches.Patch(color="red",label="Standard Deviation")
    third = mpatches.Patch(color="blue",label=f"Cooperation Rate")
    plt.legend(loc='upper right', handles=[first, third]) 

    plt.xlim(0.9*10**(-3),1.1)
    plt.title("Dynamics of the cooperation rate over different alphas")
    plt.show()
#Produces a plot of effect size and significance of mixed effect models (Link Frequency to Cooperation rate on that link) over ranging alphas
def AlphaPlot2(alphas):
    alphas=[1-n for n in alphas]
    final = pandas.read_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/Results/edgeAnalysis/AlphaAnalysis/Final.csv')
    effects = list(final.loc[:,"effect"])
    pvals = list(final.loc[:,"pval"])
    baseeff = effects[0]
    basep=pvals[0]
    #effects =[m/baseeff for m in effects]
    #pvals = [s/basep for s in pvals]
    plt.xscale("log")
    plt.xlabel("1 - Alpha")
    plt.ylabel("Effect-Size", color="blue")
    plt.tick_params(axis="y", labelcolor="blue")
    plt.plot(alphas,effects,color="blue", linestyle = "dashed")

    plt.twinx()
    plt.yscale("log")
    plt.ylabel("Significance", color = "red")
    plt.tick_params(axis="y", labelcolor="red")
    plt.plot(alphas,pvals, linestyle="dashed", color="red")


    first =mpatches.Patch(color="red",label="P-Value")
    third = mpatches.Patch(color="blue",label=f"Effect-Size")
    plt.legend(loc='upper right', handles=[first, third]) 

    plt.xlim(0.9*10**(-3),1.1)
    plt.title("Dynamics of P-Value and Effect-Size over different alphas")
    plt.show()

#Produces a plot of pValues of link between Eccentricity/Degree/Clustering to cooperation over ranging alphas
def AlphaPlot3(alphas):
    alphas=[1-n for n in alphas]
    final = pandas.read_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/Results/NodeAnalysis/Final.csv')
    eccs = list(final.loc[:,"eccentricity"])
    clusts = list(final.loc[:,"clustering"])
    degs = list(final.loc[:,"degree"])

    baseecc = eccs[0]
    basedegs = degs[0]
    baseclusts = clusts[0]

    eccs =[m/baseecc for m in eccs]
    clusts = [s/baseclusts for s in clusts]
    degs = [d/basedegs for d in degs]

    plt.xscale("log")
    plt.xlabel("1 - Alpha")
    plt.ylabel("value")
    plt.plot(alphas,clusts,color="blue", linestyle = "dashed")
    plt.plot(alphas,eccs,linestyle="dashed", color="red")
    plt.plot(alphas,degs,linestyle="dashed", color="green")


    first =mpatches.Patch(color="red",label="Normalized Significance of Eccentricity")
    second = mpatches.Patch(color="green", label="Normalized Significance of Node Degree")
    third = mpatches.Patch(color="blue",label=f"Normalized Significance of Clustering factor")
    plt.legend(loc='upper right', handles=[first, second, third]) 

    plt.xlim(0.9*10**(-3),1.1)
    plt.title("Dynamics of the significance of Eccentricity, Node Degree and Clustering factor in prediction of cooperation over different alphas")
    plt.show()

#Produces a plot effect size and significance of link between clustering/Frequency to cooperation rate over ranging alphas
def AlphaPlot4(alphas):
    alphas=[1-n for n in alphas]
    final = pandas.read_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/Results/NodeAnalysis/Final.csv')
    clustspVal = list(final.loc[:,"clustering"])
    baseclusts = clustspVal[0]
    #clustspVal = [s/baseclusts for s in clustspVal]

    clustseff = list(final.loc[:,"clusteff"])
    baseclust = clustseff[0]
    clustseff = [e / baseclust for e in clustseff]

    finaledge = pandas.read_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/Results/edgeAnalysis/AlphaAnalysis/Final.csv')
    effects = list(finaledge.loc[:,"effect"])
    pvals = list(finaledge.loc[:,"pval"])
    baseeff = effects[0]
    basep=pvals[0]
    effects =[m/baseeff for m in effects]
    #pvals = [s/basep for s in pvals]

    plt.xscale("log")
    plt.xlabel("1 - Alpha")
    plt.ylabel("Normalized Effect-Size")
    plt.tick_params(axis="y")
    plt.plot(alphas,effects,color="blue", linestyle = (0, (5, 1)))
    plt.plot(alphas, clustseff, color="red",linestyle=(0, (5, 1)))

    plt.twinx()
    plt.yscale("log")
    plt.ylabel("Significance")
    plt.tick_params(axis="y")
    plt.plot(alphas,pvals, linestyle=(0, (5, 5)), color="lightskyblue")
    plt.plot(alphas,clustspVal, linestyle=(0, (5, 5)), color="coral")

    #creat matrix legend:
    n_cols=3

    patches = [mpatches.Patch(color="white",label=""), mpatches.Patch(color="white",label='Effect Size'), mpatches.Patch(color="white", label='P-Value')]
    first =mpatches.Patch(color="red",label="------------------")
    second = mpatches.Patch(color="coral", label="- - - - - - - - - -")
    third = mpatches.Patch(color="blue",label="------------------")
    fourth = mpatches.Patch(color="lightskyblue", label="- - - - - - - - - -")
    patches.append(mpatches.Patch(color="white",label="Frequency"))
    patches.append(third)
    patches.append(fourth)
    patches.append(mpatches.Patch(color="white", label="Clustering Factor"))
    patches.append(first)
    patches.append(second)

    
    cols=["black", "black", "black","black","blue", "lightskyblue", "black","red","coral"]
    l = plt.legend(bbox_to_anchor=(0.35,0.8),ncol=n_cols, handles=patches) 
    count=0
    for text in l.get_texts():
        text.set_color(cols[count])
        count+=1

    plt.xlim(0.9*10**(-3),1.1)
    plt.title("Dynamics of P-Value and Effect Size of the relationship between Frequency/Clustering Factor and cooperation over different alphas")
    plt.show()

#Produces linear Regression between presence of different rules and cooperation levels
def RuleReg(means,combs):
    binKS = [0]*16
    binIR = [0]*16
    binDR = [0]*16 
    binN = [0]*16
    c = 0
    for comb in combs:
        for rule in comb:
            if rule == "Net":
                binN[c]=1
            elif rule == "IR":
                binIR[c] = 1
            elif rule == "DR":
                binDR[c] = 1
            elif rule == "KS":
                binKS[c] = 1
        c+=1
    df = pandas.DataFrame({'coop': means, 'Net': binN, 'IR': binIR, 'DR': binDR, 'KS': binKS})
    df.to_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/DiffRules/linregdata.csv')
    

    pass




flatresults=[]
for j in range(200):
    csvFile = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/coopstats{j}.csv')
    col = len(csvFile.columns)
    csvFile = csvFile.iloc[:,100:col]
    x=csvFile.values.tolist()
    flatresults.append([item for sublist in x for item in sublist])
alphas = [0.999, 0.997, 0.995, 0.993, 0.99, 0.97, 0.95, 0.93, 0.9, 0.7, 0.5, 0.3, 0]
first=[item for sublist in flatresults[0:99] for item in sublist]
second = [item for sublist in flatresults[100:200] for item in sublist]

flatresults = [first, second]
print(ttest_ind(flatresults[0], flatresults[1],equal_var=False))


sns.set(style="darkgrid")
cols=["blue", "green", "coral", "mediumpurple", "red", "mediumpurple", "aquamarine", "indigo", "palegreen", "yellow"]

lines = 'Statistical Results:'
with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/Stats.txt', 'w') as f:
    f.write(lines)

rules = ["Net","IR","DR","KS"]
combs=[[]]
for l in range(4):
    combs +=list(combinations(rules,l+1))
for q in range(len(combs)):
    combs[q]=list(combs[q])

counter=0
means=[]
stds=[]
for res in flatresults:
    if counter==0:
        col = "red"
    elif counter == 1:
        col="blue"
    else:
        col="green"
    sns.histplot(data=res, x=res, kde=True, bins = 100, stat='density', color=col)#, label=f'Rules = {combs[counter]}')

    mean=statistics.mean(res)
    var=statistics.stdev(res)
    means.append(mean)
    stds.append(var)
    more_lines=[f"Run number {counter}", f"Mean = {mean}", f"Standard deviation = {var}"]
    with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/stats.txt', 'a') as f:
        for line in more_lines:
            f.write(line)
            f.write('\n')
    counter+=1
first =mpatches.Patch(color="red",label=f"Randomized Agta Network")
second = mpatches.Patch(color="blue",label=f"Randomized Agta Networks with Direct Reciprocity")
third = mpatches.Patch(color="green", label =f"Multilevel Agta Network with Kin Selection")
plt.xlabel("Cooperation rate")
plt.ylabel("Density")
plt.xlim(0,1)
plt.legend(loc='upper right', handles=[first, second]) 
plt.title("Impact of Kin Selection")
plt.show()
#AlphaPlot(alphas,means,stds)
