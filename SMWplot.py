import networkx as nx
import random
from matplotlib import pyplot as plt
import statistics
import matplotlib.patches as mpatches

probs = [0.0001, 0.000175, 0.0003, 0.0005, 0.0007, 0.001, 0.00175, 0.003, 0.005, 0.007, 0.01, 0.0175, 0.03, 0.05, 0.07, 0.1, 0.175, 0.3, 0.5, 0.7, 1]
means =[0.44290366731354053, 0.4439000408146504, 0.44389802374195336, 0.44013996565316316, 0.43997526773473916, 0.43923909655105436, 0.4452850950255272, 0.4424690780283019, 0.44992542586681467,0.4471048195527192, 0.44459672296892344, 0.44848805240122086, 0.4465248830493896, 0.44790322900665924, 0.4467139693801332, 0.43383949674028854, 0.42222031007880134, 0.37533930381021086,0.34771394067702555,0.3452602100965594, 0.3446019909944506]

#probs = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1]
#means =[0.451, 0.444, 0.450, 0.438, 0.447, 0.445, 0.443, 0.445, 0.451, 0.444, 0.446, 0.434, 0.445, 0.375, 0.352, 0.342, 0.346]
means1 = [0.439, 0.427, 0.452, 0.434, 0.430, 0.436, 0.423, 0.438, 0.441, 0.439, 0.442, 0.445, 0.439, 0.374, 0.357, 0.345, 0.343]
stds = [0.039,0.047,0.038,0.045,0.045,0.051,0.041,0.041,0.042,0.048,0.044,0.051,0.054,0.050,0.056,0.049,0.051]
'''
i=0
for p in probs:
    F = nx.watts_strogatz_graph(1000, 10, p)
    nx.write_weighted_edgelist(F, f"C:/Users/klaus/Documents/Uni/Masterarbeit/Project 1/SMWnets/smw{i}.txt")
    i+=1
quit()
'''

baseline=nx.watts_strogatz_graph(1000,10,0)
baseclus=nx.average_clustering(baseline)
baseaspl=nx.average_shortest_path_length(baseline)
basecoop=0.44513290036514985

means=[q/basecoop for q in means]

clusts=[0.9995670454545456, 0.9994763636363637, 0.9991421212121213, 0.9982009848484847, 0.9979565909090908, 0.9964759848484848, 0.9942536363636358, 0.9912302056277054, 0.9847309415584409, 0.9802582334332332, 0.9731290692640681, 0.952007510822511, 0.9190242865467879, 0.867956331168835, 0.8107753275890812, 0.7353385289710328, 0.5678086352128294, 0.3555031682533172, 0.1385178445715803, 0.03893718208777836, 0.013513515009242628]
aspl=[0.9199196865079364, 0.8970646984126984, 0.8172330952380952, 0.6994892142857142, 0.6343497658730158, 0.5221122301587301, 0.40094295238095234, 0.3072399087301587, 0.23975603174603172, 0.21038814285714286, 0.18319463095238092, 0.14398524999999998, 0.12093490079365078, 0.10502496825396825, 0.09558383333333331, 0.087826623015873, 0.07848749206349206, 0.07156813888888888, 0.06707465476190476, 0.06524645634920635, 0.06477581349206349]

plt.xscale("log")
plt.ylim(0.75,1.05)
plt.xlabel("Rewiring Probability (p)")
plt.ylabel("Cooperation Rate", color="blue")
plt.tick_params(axis="y", labelcolor="blue")
plt.plot(probs,means,color="blue", linestyle = "dashed")

plt.twinx()
plt.ylabel("C(p)/C(0) \nL(p)/L(0)")
plt.scatter(y=clusts,x=probs, marker="s", color="red")
plt.scatter(y=aspl, x=probs, color="green")


first =mpatches.Patch(color="red",label="Normalized Clustering Coefficient")
second = mpatches.Patch(color="green",label=f"Normalized Average Shortest Path Length")
third = mpatches.Patch(color="blue",label=f"Normalized Cooperation Rate")
plt.legend(loc='upper right', handles=[first, second, third]) 

plt.xlim(0.9*10**(-4),1.1)
plt.title("Dynamics of the cooperation rate over the Small-World spectrum")
plt.show()

quit()
nets=[]
for i in range(21):
    temp=[]
    for j in range(10):
        G = nx.watts_strogatz_graph(1000,10,probs[i])
        temp.append(G)
    nets.append(temp)
print("generation of networks done")
clusts=[]
aspl=[]
for net in nets:
    tempclus=[]
    tempaspl=[]
    for acnet in net:
        tempclus.append(nx.average_clustering(acnet))
        tempaspl.append(nx.average_shortest_path_length(acnet))
    clusts.append(statistics.mean(tempclus)/baseclus)
    aspl.append(statistics.mean(tempaspl)/baseaspl)
print(f"clustering:{clusts}")
print(f"average shortest path length: {aspl}")
plt.plot(means)
plt.plot(clusts)
plt.plot(aspl)
plt.grid()
plt.show()

quit()