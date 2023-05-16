from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
from matplotlib.ticker import MultipleLocator
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42

avg_SPEB = pkl.load(open("avg_SPEB.pkl", "rb"))
avg_SPEB_md = pkl.load(open("avg_SPEB_md.pkl", "rb"))
avg_allSPEB = pkl.load(open("avg_allSPEB.pkl", "rb"))
avg_SPEB_ep = pkl.load(open("avg_SPEB_ep.pkl", "rb"))

print(np.sum(np.log10(np.sum(avg_SPEB, axis=1))-np.log10(np.sum(avg_SPEB_ep, axis=1))))
print(avg_SPEB)
print(avg_SPEB_md[3])
print(avg_SPEB_md[10])
print(avg_allSPEB)
print(avg_SPEB_ep)

ax = plt.subplot(111)
plt.plot(np.arange(1, 11) * 0.1, 10 * np.log10(np.sum(avg_SPEB_md[3], axis=1) / 10), label="distance method (3 anchors)",
         marker="^", markersize=10, linestyle=":", linewidth=3,color="orange")
plt.plot(np.arange(1, 11) * 0.1, 10 * np.log10(np.sum(avg_allSPEB, axis=1) / 10), label="all anchors method",
         marker="s",markersize=10, linestyle="--", linewidth=3,color="green")
plt.plot(np.arange(1, 11) * 0.1, 10 * np.log10(np.sum(avg_SPEB_md[10], axis=1) / 10), label="distance method (10 anchors)",
         marker="v", markersize=10, linestyle="-.", linewidth=3,color="firebrick")
plt.plot(np.arange(1, 11) * 0.1, 10 * np.log10(np.sum(avg_SPEB_ep, axis=1) / 10), label="proposed MLE method",
         marker="D",markersize=10, linestyle="-", linewidth=3,color="purple")
plt.plot(np.arange(1, 11) * 0.1, 10 * np.log10(np.sum(avg_SPEB, axis=1) / 10), label="proposed greedy method", marker="o",
         markersize=10, linewidth=3)
plt.grid()

xminorLocator = MultipleLocator(0.04)
yminorLocator = MultipleLocator(2)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel("bandwidth(GHz)", fontsize=24)
plt.ylabel("average $\mathrm{SPEB_{pr}}\ (\mathrm{dB},\mathrm{m}^2)$", fontsize=24)
plt.legend(fontsize=20)
plt.show()