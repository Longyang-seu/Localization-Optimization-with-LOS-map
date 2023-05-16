from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
plt.rcParams["font.family"]=["sans-serif"]
plt.rcParams["font.sans-serif"]=["SimSun"]

SPEB = pkl.load(open("beam_SPEB_NT_sig.pkl", "rb"))
print(SPEB)

plt.plot(range(2,12,2), 10*np.log10(SPEB[1]), label="基于位置的单锚点方法", linestyle="-.", marker="^",
         markersize=10, linewidth=3)
plt.plot(range(2,12,2), 10*np.log10(SPEB[2]),label="基于位置的双锚点覆盖方法", marker="o",markersize=10, linewidth=3)
plt.plot(range(2,12,2), 10*np.log10(SPEB[0]),label="所提出方法", marker="x",markersize=10, linewidth=3)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel("用户先验分布的标准差（m）", fontsize=24)
plt.ylabel("${\mathrm{SPEB}}_{\mathrm{pr},\mathbf{P}}(\mathbf{f}_{\mathrm{opt}})$,dB $\mathrm{m}^2$", fontsize=24)
plt.legend(fontsize=20)
plt.show()