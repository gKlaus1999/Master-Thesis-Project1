import pandas
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
import statistics
from itertools import combinations
from scipy.stats import ttest_ind
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Produces a Plot of means and Standard deviation of cooperation over different alphas
def AlphaPlot(alphas, means, stds):
    sns.reset_defaults()
    alphas=[1-n for n in alphas]
    basemean=means[0]   #get base mean and std to possibly standardize results
    basestd=stds[0]
    #means =[m/basemean for m in means]
    #stds = [s/basestd for s in stds]
    plt.xscale("log")   #turn x axis to log
    plt.xlabel("1 - Alpha")
    plt.ylabel("Cooperation Rate", color="blue")    
    plt.tick_params(axis="y", labelcolor="blue")
    plt.plot(alphas,means,color="blue", linestyle = "dashed")

    #get second yaxis on the right
    plt.twinx()
    plt.ylabel("Standard Deviation", color = "red")
    plt.tick_params(axis="y", labelcolor="red")
    plt.plot(alphas,stds, linestyle="dashed", color="red")


    first =mpatches.Patch(color="red",label="Standard Deviation")
    third = mpatches.Patch(color="blue",label=f"Cooperation Rate")
    plt.legend(loc='upper right', handles=[first, third]) 

    plt.xlim(0.9*10**(-2),1.1)
    plt.title("Dynamics of the cooperation rate over different alphas")
    plt.show()

#Produces a plot of effect size and significance of mixed effect models (Link Frequency to Cooperation rate on that link) over ranging alphas
def AlphaPlot2(alphas):
    alphas=[1-n for n in alphas]

    finalRan = pandas.read_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/RanAlphaPlot/FinalEdge.csv')
    '''final = pandas.read_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/FinalEdge.csv')
    effects = list(final.loc[:,"effect"])
    pvals = list(final.loc[:,"pval"])
    ints = list(final.loc[:,"intercept"])
    coopsAg = [0.468,0.447,0.404,0.422,0.404,0.327,0.306,0.285,0.258]
    
    effects =[m/min(effects) for m in effects]
    ints = [m/min(ints) for m in ints]
    coopsAg = [m/min(coopsAg) for m in coopsAg]'''

    effectsRan = list(finalRan.loc[:,"effect"])
    pvalsRan = list(finalRan.loc[:,"pval"])
    intsRan = list(finalRan.loc[:,"intercept"])
    coopsRan = [0.354,0.329,0.315,0.293,0.287,0.231,0.209,0.192,0.168]

    effectsRan = [m/min(effectsRan)for m in effectsRan]
    intsRan = [m/min(intsRan)for m in intsRan]
    coopsRan = [m/min(coopsRan) for m in coopsRan]
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Size")
    #plt.tick_params(axis="y", labelcolor="blue")
    '''plt.plot(alphas, coopsAg, color="blue", linestyle = "dashed")
    plt.plot(alphas,effects,color="red", linestyle = "dashed")
    plt.plot(alphas,ints, linestyle="dashed", color="green")'''


    plt.plot(alphas,coopsRan,color="blue", linestyle = "dashed")
    plt.plot(alphas,effectsRan,color="red", linestyle = "dashed")
    plt.plot(alphas,intsRan, linestyle="dashed", color="lightblue")
    

    first =mpatches.Patch(color="blue",label="Normalized cooperation rate")
    sec = mpatches.Patch(color="red", label = "Normalized effect-size of F")
    third = mpatches.Patch(color="lightblue",label=f"Normalized Y-Intercept")
    plt.legend(loc='upper left', handles=[first, sec,third]) 

    plt.xlim(0.9*10**(-2),1.1)
    plt.gca().invert_xaxis()
    plt.title("Y-Intercepts and Effect size of F on cooperation over different alphas")
    plt.show()

#Produces plot with significance of R and F over kinship networks with different correlations
def CorrPlot():
    final = pandas.read_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/DiffCorr/corrsFin.csv')
    Reff = list(final.loc[:,"R-Effect"])
    Feff = list(final.loc[:,"F-Effect"])
    ints = list(final.loc[:, "intercept"])
    corrs = list(final.loc[:, "corrs"])
    plt.xlabel("Correlation between kinship- and social network")
    plt.ylabel("Effect-Size of R")
    plt.plot(corrs,Reff, color="darkblue", linestyle = "dashed")

    plt.twinx()
    plt.ylabel("Effect-Size of F", color = "red")
    plt.tick_params(axis="y", labelcolor="red")
    plt.plot(corrs,Feff, linestyle = "dashed", color = "red")

    first=mpatches.Patch(color="blue", label="Effect-Size of R")
    second=mpatches.Patch(color="red", label="Effect-Size of F")
    plt.legend(loc="upper left", handles=[first, second])
    plt.title("Effect-size of R and F in different kinship networks")
    plt.show()

    nReff = [eff/max(Reff) for eff in Reff]
    nInts = [int/max(ints) for int in ints]
    nFeff = [eff/max(Feff) for eff in Feff]
    plt.xlabel("Correlation between kinship- and social network")
    plt.ylabel("Normalized Size")
    plt.plot(corrs, nReff, color = "blue", linestyle = "dashed")
    plt.plot(corrs, nFeff, color = "red", linestyle = "dashed")
    plt.plot(corrs, nInts, color = "lightblue", linestyle = "dashed")
    first = mpatches.Patch(color="blue", label = "Effect-Size of R")
    sec = mpatches.Patch(color="red", label="Effect-Size of F") 
    third = mpatches.Patch(color="lightblue", label="Y-Intercept")
    plt.legend(loc="lower right", handles=[first, sec, third])
    plt.title("Y-Intercept, Effect-size of R and F in different kinship networks")
    plt.show()

#Produces a plot of pValues of link between Eccentricity/Degree/Clustering to cooperation over ranging alphas
def AlphaPlot3(alphas):
    alphas=[1-n for n in alphas]
    final = pandas.read_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/FinalNode.csv')
    #eccs = list(final.loc[:,"eccentricity"])
    clusts = list(final.loc[:,"clusteff"])
    #degs = list(final.loc[:,"degree"])

    #baseecc = eccs[0]
    #basedegs = degs[0]
    #baseclusts = clusts[0]

    #eccs =[m/baseecc for m in eccs]
    #clusts = [s/baseclusts for s in clusts]
    #degs = [d/basedegs for d in degs]

    plt.xscale("log")
    plt.xlabel("1 - Alpha")
    plt.ylabel("value")
    plt.plot(alphas,clusts,color="blue", linestyle = "dashed")
    #plt.plot(alphas,eccs,linestyle="dashed", color="red")
    #plt.plot(alphas,degs,linestyle="dashed", color="green")


    #first =mpatches.Patch(color="red",label="Normalized Significance of Eccentricity")
    #second = mpatches.Patch(color="green", label="Normalized Significance of Node Degree")
    third = mpatches.Patch(color="blue",label=f"Effect Size of Clustering factor")
    plt.legend(loc='upper right', handles=[third]) 

    plt.xlim(0.9*10**(-2),1.1)
    plt.title("Effect size of Clustering factor for the prediction of cooperation over different alphas")
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

#Produces data for linear Regression between presence of different rules and cooperation levels
def RuleReg(means,combs):
    binKS = [0]*16
    binIR = [0]*16
    binDR = [0]*16 
    binN = [0]*16
    #binSMW = [0]*24
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
    df = pandas.DataFrame({'coop': means,'std': stds, 'Net': binN, 'IR': binIR, 'DR': binDR, 'KS': binKS})
    df.to_csv('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/linregdata.csv')
    

    pass



alphas = [0.99, 0.97, 0.95, 0.93, 0.9, 0.7, 0.5, 0.3, 0]

#AlphaPlot2(alphas)

results = 10     #number of different simulation runs of interest
replicas = 1    #number of replicas (100 different networks)
rules = ["Net","SMW","IR","DR","KS"]
rules = ["Net","IR","DR","KS"]
#generate all possible combinations of rules 
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

#Turn all the results into a list of lists with every interior list being all the coop values from all the replicas
flatresults=[]
for e in range(21):
    replicas = 100
    for j in range(replicas):
        csvFile = pandas.read_csv(F'C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/coopstats{e, j}.csv')
        col = len(csvFile.columns)
        csvFile = csvFile.iloc[:,100:col]       #cut off first 100 generations until simulation has reached equilibrium
        x=csvFile.values.tolist()
        flatresults.append([item for sublist in x for item in sublist])
        #If replicas exist put them into one list and delete the replicas
        if replicas>1:
            flatresults[e] = [item for sublist in flatresults[e:e+replicas-1] for item in sublist]
            del flatresults[e+1:e+replicas-1]
print(len(flatresults))
#write down results (mean and std) in txt file
lines = 'Statistical Results:'
with open('C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/plots/Stats.txt', 'w') as f:
    f.write(lines)

counter=0
means=[]
stds=[]
print(len(flatresults))
sns.set(style="darkgrid")   #set scheme of plot
cols=iter(cm.rainbow(np.linspace(0,1,results))) #generate colors for the plot
for res in flatresults: #go through every simulation
    #sns.histplot(data=res, x=res, kde=True, bins = 100, stat='density', color=next(cols), label=f'Rules : {combs[counter]}')    #make a density plot

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
#alphas = [0.99, 0.97, 0.95, 0.93, 0.9, 0.7, 0.5, 0.3, 0]
cols=iter(cm.rainbow(np.linspace(0,1,results)))
handles=[]
handles.append(mpatches.Patch(color=next(cols),label=f"randomized Agta Network"))
handles.append(mpatches.Patch(color=next(cols),label=f"Indirect Reciprocity in the randomized Agta Network"))
'''handles.append(mpatches.Patch(color=next(cols), label =f"Randomized Agta Networks"))
handles.append(mpatches.Patch(color=next(cols), label =f"Agta Network"))
handles.append(mpatches.Patch(color=next(cols), label =f"+1 std"))
handles.append(mpatches.Patch(color=next(cols), label =f"control"))
handles.append(mpatches.Patch(color=next(cols), label =f"Alpha = 0.5"))
handles.append(mpatches.Patch(color=next(cols), label =f"Alpha = 0.3"))
handles.append(mpatches.Patch(color=next(cols), label =f"Alpha = 0"))'''
plt.xlabel("Cooperation rate")
plt.ylabel("Density")
plt.xlim(0,1)
plt.legend(loc='upper right')#,handles=handles) 
plt.title("Impact of Direct Reciprocity on the Agta Network")
#plt.show()
#RuleReg(means,combs)
quit()
corrs=[0.19688275483286632,0.19688275483286632,0.19688275483286632,0.1493723423515148,0.1462806448400385,0.14258732103928545,0.09137140131997608,0.09161839969157441,0.099652444225055,0.04599636119484679,0.046951741514114124,0.04864485664807418,-0.0013014503294101227,-0.00013275120741330706,-0.010600967955944158,-0.061978136469002576,-0.05850110356834506,-0.05257380639095638,-0.10343388017426648,-0.10002890426579325,-0.10026832278212165]
corrs=np.array(corrs).reshape(-1,1)
means=np.array(means).reshape(-1,1)
means = means-0.2575369124358914
model = LinearRegression()
model.fit(corrs,means)
print(model.intercept_, model.coef_, model.score(corrs,means))

plt.clf()
plt.scatter(corrs,means)
plt.plot(corrs,model.predict(corrs), color="red", label = "R-Sq = 0.743")
plt.ylabel("Impact of Kin Selection")
plt.xlabel("R-F Correlation")
plt.title("KS Impact = 0.138 + 0.144 Corr")
plt.legend(loc="upper right")
plt.show()