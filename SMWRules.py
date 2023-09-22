import networkx as nx
import random
from matplotlib import pyplot as plt
import statistics
import matplotlib.patches as mpatches

probs = [0.0001, 0.000175, 0.0003, 0.0005, 0.0007, 0.001, 0.00175, 0.003, 0.005, 0.007, 0.01, 0.0175, 0.03, 0.05, 0.07, 0.1, 0.175, 0.3, 0.5, 0.7, 1]
means =[0.19025261845386535, 0.18947605985037408, 0.1870713216957606, 0.18954039900249375, 0.19515336658354115, 0.1947201995012469, 0.19148877805486283, 0.18964014962593517, 0.18930399002493767, 0.1920927680798005, 0.1920852867830424, 0.1930932668329177, 0.18299476309226934, 0.187328927680798, 0.18950349127182045, 0.18862369077306734, 0.1794074812967581, 0.18067381546134664, 0.17554862842892768, 0.17054189526184538, 0.17082319201995014]
meansIR =[0.37399152119700746, 0.37302643391521195, 0.37735236907730674, 0.3672199501246883, 0.37116134663341643, 0.3717216957605985, 0.36887057356608477, 0.37651945137157106, 0.3619219451371571, 0.366798753117207, 0.36646259351620947, 0.36646259351620947, 0.3601526184538653, 0.3576246882793018, 0.35837456359102243, 0.3623319201995012, 0.3499234413965087, 0.33216658354114714, 0.31148204488778053, 0.3038441396508728, 0.30157556109725686, 0.28922643391521197]
meansDR = [0.39414223715461344, 0.39141503859102245, 0.3981007217805486, 0.38499203406982546, 0.3807889966184539, 0.37757904121446384, 0.36787655325935165, 0.37535857401745637, 0.38340701343890277, 0.3824283614763092, 0.41062514534413963, 0.385916171286783, 0.3613163577805486, 0.36679372825935164, 0.3632858382743142, 0.3706971745810474, 0.3359842753241895, 0.30901630510972566, 0.27477787250374064, 0.2754458838703242, 0.27228380667830426]
meansKS = [0.3672596009975062, 0.3684077306733167, 0.3620134663341646, 0.3646311720698254, 0.3691648379052369, 0.3657992518703242, 0.36678952618453864, 0.36875236907730674, 0.3618167082294264, 0.36449800498753115, 0.3625234413965087, 0.3715007481296758, 0.3614428927680798, 0.36165810473815463, 0.3666221945137157, 0.36181945137157107, 0.3589336658354115, 0.36383790523690773, 0.3620221945137157, 0.35558603491271823, 0.3600386533665835]

diffIR = [meansIR[k]-means[k] for k in range(len(means))]
diffDR = [meansDR[k]-means[k] for k in range(len(means))]
diffKS = [meansKS[k]-means[k] for k in range(len(means))]

baseline=nx.watts_strogatz_graph(1000,10,0)
baseclus=nx.average_clustering(baseline)
baseaspl=nx.average_shortest_path_length(baseline)
baseIR = diffIR[0]
baseDR = diffDR[0]
baseKS = diffKS[0]
print(diffKS)

diffIR=[q/baseIR for q in diffIR]
diffDR=[q/baseDR for q in diffDR]
diffKS=[q/baseKS for q in diffKS]
print(diffKS)

clusts=[0.9995670454545456, 0.9994763636363637, 0.9991421212121213, 0.9982009848484847, 0.9979565909090908, 0.9964759848484848, 0.9942536363636358, 0.9912302056277054, 0.9847309415584409, 0.9802582334332332, 0.9731290692640681, 0.952007510822511, 0.9190242865467879, 0.867956331168835, 0.8107753275890812, 0.7353385289710328, 0.5678086352128294, 0.3555031682533172, 0.1385178445715803, 0.03893718208777836, 0.013513515009242628]
aspl=[0.9199196865079364, 0.8970646984126984, 0.8172330952380952, 0.6994892142857142, 0.6343497658730158, 0.5221122301587301, 0.40094295238095234, 0.3072399087301587, 0.23975603174603172, 0.21038814285714286, 0.18319463095238092, 0.14398524999999998, 0.12093490079365078, 0.10502496825396825, 0.09558383333333331, 0.087826623015873, 0.07848749206349206, 0.07156813888888888, 0.06707465476190476, 0.06524645634920635, 0.06477581349206349]

plt.xscale("log")
plt.ylim(0.3,1.25)
plt.xlabel("Rewiring Probability (p)")
plt.ylabel("Normalized impact on Cooperation", color="blue")
plt.tick_params(axis="y", labelcolor="blue")

plt.plot(probs,diffIR,color="darkblue", linestyle = "dashed")
plt.plot(probs,diffDR,color="c", linestyle = "dashed")
plt.plot(probs, diffKS, color="lightblue", linestyle="dashed")

plt.twinx()
plt.ylabel("C(p)/C(0) \nL(p)/L(0)")
plt.scatter(y=clusts,x=probs, marker="s", color="darkred")
plt.scatter(y=aspl, x=probs, color="orange")


first =mpatches.Patch(color="darkred",label="Normalized Clustering Coefficient")
second = mpatches.Patch(color="orange",label=f"Normalized Average Shortest Path Length")
third = mpatches.Patch(color="darkblue",label=f"Normalized impact of Indirect Reciprocity")
fourth = mpatches.Patch(color="c",label=f"Normalized impact of Direct Reciprocity")
fifth = mpatches.Patch(color ="lightblue", label="Impact of Kin Selection")
plt.legend(loc='upper right', handles=[first, second, third, fourth,fifth]) 

plt.xlim(0.9*10**(-4),1.1)
plt.title("Dynamics of the impact of Direct Reciprocity, Indirect Reciprocity, Kin Selection on the cooperation rate over the Small-World spectrum")
plt.show()

