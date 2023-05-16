from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
plt.rcParams["font.family"]=["sans-serif"]
plt.rcParams["font.sans-serif"]=["SimSun"]

SPEB0 = pkl.load(open("beam_SPEB_NT.pkl", "rb"))
SPEB1 = pkl.load(open("beam_SPEB_NT_more1.pkl", "rb"))
SPEB_dis1 = pkl.load(open("beam_SPEB_2dis_NT1.pkl","rb"))
SPEB_dis2 = pkl.load(open("beam_SPEB_2dis_NT2.pkl","rb"))
print(SPEB0,SPEB1)
SPEB = np.concatenate((SPEB0,SPEB1),axis=1)
SPEB_dis = np.concatenate((SPEB_dis1, SPEB_dis2),axis=1).reshape(-1)
print(SPEB)

plt.plot(range(16,112,16), 10*np.log10(SPEB[1]), label="基于位置的单锚点方法", linestyle="-.", marker="^",
         markersize=10, linewidth=3)
plt.plot(range(16,112,16), 10*np.log10(SPEB_dis),label="基于位置的双锚点覆盖方法", marker="o",markersize=10, linewidth=3)
plt.plot(range(16,112,16), 10*np.log10(SPEB[0]),label="所提方法", marker="x",markersize=10, linewidth=3)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel("天线数量", fontsize=24)
plt.ylabel("${\mathrm{SPEB}}_{\mathrm{pr},\mathbf{P}}(\mathbf{f}_{\mathrm{opt}})$,dB $\mathrm{m}^2$", fontsize=24)
plt.legend(fontsize=20)
plt.show()