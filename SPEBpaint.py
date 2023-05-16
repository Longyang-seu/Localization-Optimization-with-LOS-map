from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42

a_num=36

SPEB = pkl.load(open("SPEB.pkl", "rb"))
SPEB_md = pkl.load(open("SPEB_md.pkl", "rb"))
SPEB_ep = pkl.load(open("SPEB_ep.pkl", "rb"))
val=len(SPEB_ep)
for i in range(a_num-val+1):
    SPEB_ep.append(SPEB_ep[val-1])
print(10 * np.log10(SPEB))
print(10 * np.log10(SPEB_md))
print(10 * np.log10(SPEB_ep))

ax = plt.subplot(111)
plt.hlines(y=10 * min(np.log10(SPEB)),xmin=1.5,xmax=37,colors="r",linestyles=":")
plt.hlines(y=10 * min(np.log10(SPEB_ep)),xmin=2,xmax=4,colors="r",linestyles=":")
plt.hlines(y=10 * min(np.log10(SPEB_md)),xmin=7,xmax=9,colors="r",linestyles=":")
plt.hlines(y=10 * np.log10(SPEB)[36],xmin=35,xmax=37,colors="r",linestyles=":")
plt.vlines(x=3,ymax=10 * min(np.log10(SPEB_ep)),ymin=10 * min(np.log10(SPEB)),colors="r")
plt.vlines(x=8,ymax=10 * min(np.log10(SPEB_md)),ymin=10 * min(np.log10(SPEB)),colors="r")
plt.vlines(x=36,ymax=10 * np.log10(SPEB_md)[36],ymin=10 * min(np.log10(SPEB)),colors="r")
plt.plot(range(a_num + 1), 10 * np.log10(SPEB_md), label="distance method", linestyle="-.", marker="^",
         markersize=10, linewidth=3, color="orange")
plt.plot(range(8), 10 * np.log10(SPEB_ep[:8]),label="proposed MLE method", marker="x",markersize=10, linewidth=3, color="purple")
plt.plot(range(a_num + 1), 10 * np.log10(SPEB), label="proposed greedy method", marker=".", markersize=10, linewidth=3)
plt.text(3,5*(min(np.log10(SPEB))+min(np.log10(SPEB_ep))),"%.2f dB" % abs(float(10*(min(np.log10(SPEB))-min(np.log10(SPEB_ep))))),fontsize=20,ha="left")
plt.text(8,5*(min(np.log10(SPEB))+min(np.log10(SPEB_md))),"%.2f dB" % abs(float(10*(min(np.log10(SPEB))-min(np.log10(SPEB_md))))),fontsize=20,ha="left")
plt.text(36,5*(min(np.log10(SPEB))+np.log10(SPEB[36])),"%.2f dB" % abs(float(10*(min(np.log10(SPEB))-np.log10(SPEB[36])))),fontsize=20,ha="right")
plt.grid()

xminorLocator = MultipleLocator(1)
yminorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel("number of selected anchors", fontsize=24)
plt.ylabel("$\mathrm{SPEB_{pr}}\ (\mathrm{dB},\mathrm{m}^2)$", fontsize=24)
plt.legend(fontsize=20)
plt.show()