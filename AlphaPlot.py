import random
from matplotlib import pyplot as plt
import statistics
import matplotlib.patches as mpatches

alphas = [0.999, 0.997, 0.995, 0.993, 0.99, 0.97, 0.95, 0.93, 0.9, 0.7, 0.5, 0.3, 0]
alphas = [1-n for n in alphas]
means = [0.527,0.502,0.526,0.479,0.508,0.506,0.475,0.476,0.433,0.383,0.373,0.325,0.314]
stds = [0.171,0.172,0.179,0.182,0.147,0.174,0.179,0.166,0.159,0.155,0.144,0.132,0.144]

plt.xscale("log")
plt.ylim(0.3,0.6)
plt.xlabel("1-Alpha")
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
