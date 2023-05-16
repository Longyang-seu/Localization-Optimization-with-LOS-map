import pickle as pkl
import scipy.integrate
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42

# system parameter
sig = 7.5
sig_num = 3
B_s = 100e6
N_0 = 1e-20
c = 3e8
f = 4.5e9
Es = 1e-11
a_num = 36
Lamb = 8 * np.pi ** 2 * B_s ** 2 * Es / (3 * c ** 2 * N_0)


def jud_LOS(map, q, ob):
    LOS_map = np.ones((a_num, int(20 * sig_num * sig), int(20 * sig_num * sig)))
    for n in range(a_num):
        for i in range(int(20 * sig_num * sig)):
            for j in range(int(20 * sig_num * sig)):
                print(n * int(20 * sig_num * sig) ** 2 + i * int(20 * sig_num * sig) + j, "/",
                      a_num * int(20 * sig_num * sig) ** 2)
                for m in range(ob.shape[0]):
                    if map[i, j, 0] < ob[m, 0] < q[n, 0] or q[n, 0] < ob[m, 0] < map[i, j, 0]:
                        if ob[m, 1] < (map[i, j, 1] - q[n, 1]) / (map[i, j, 0] - q[n, 0]) * (ob[m, 0] - map[i, j, 0]) \
                                + map[i, j, 1] < ob[m, 1] + ob[m, 3]:
                            LOS_map[n, i, j] = 0
                            break
                    if map[i, j, 0] < ob[m, 0] + ob[m, 2] < q[n, 0] or q[n, 0] < ob[m, 0] + ob[m, 2] < map[
                        i, j, 0]:
                        if ob[m, 1] < (map[i, j, 1] - q[n, 1]) / (map[i, j, 0] - q[n, 0]) * (
                                ob[m, 0] + ob[m, 2] - map[i, j, 0]) \
                                + map[i, j, 1] < ob[m, 1] + ob[m, 3]:
                            LOS_map[n, i, j] = 0
                            break
                    if map[i, j, 1] < ob[m, 1] < q[n, 1] or q[n, 1] < ob[m, 1] < map[i, j, 1]:
                        if ob[m, 0] < (map[i, j, 0] - q[n, 0]) / (map[i, j, 1] - q[n, 1]) * (ob[m, 1] - map[i, j, 1]) \
                                + map[i, j, 0] < ob[m, 0] + ob[m, 2]:
                            LOS_map[n, i, j] = 0
                            break
                    if map[i, j, 1] < ob[m, 1] + ob[m, 3] < q[n, 1] or q[n, 1] < ob[m, 1] + ob[m, 3] < map[
                        i, j, 1]:
                        if ob[m, 0] < (map[i, j, 0] - q[n, 0]) / (map[i, j, 1] - q[n, 1]) * (
                                ob[m, 1] + ob[m, 3] - map[i, j, 1]) \
                                + map[i, j, 0] < ob[m, 0] + ob[m, 2]:
                            LOS_map[n, i, j] = 0
                            break
    return LOS_map


def get_a_map(map, q):
    a = np.zeros((a_num, int(20 * sig_num * sig), int(20 * sig_num * sig)))
    for i in range(a_num):
        a[i, :] = (c / (4 * np.pi * f)) ** 2 / ((map[:, :, 0] - q[i, 0]) ** 2 + (map[:, :, 1] - q[i, 1]) ** 2)
    return a

def get_rho(LOS_map, a_map, map):
    return np.sum(np.sum(map * 1 / (LOS_map * Lamb * a_map * sig ** 2 + 1), axis=1), axis=1)


def get_phi(p, q):
    return np.arctan((p[:, :, 1] - q[1]) / (p[:, :, 0] - q[0]))


if __name__ == "__main__":
    # set the size and position of the obstacle
    ob = pkl.load(open("ob.pkl", "rb"))
    q = pkl.load(open("q.pkl", "rb"))
    p = pkl.load(open("p.pkl", "rb"))
    p_a = [500, 500]

    # Distance of anchors
    dis_q = np.zeros((a_num, 2))
    for i in range(a_num):
        dis_q[i, 1] = i
        dis_q[i, 0] = np.sqrt((q[i, 0]-p_a[0]) ** 2 + (q[i, 1]-p_a[1]) ** 2)
    dis_q = dis_q[np.lexsort(dis_q[:, ::-1].T)]
    print(dis_q)

    # Draw position diagram
    plt.figure(figsize=(10, 10), dpi=100)
    plt.scatter(q[..., 0], q[..., 1], c="b", marker="o", label="anchor position")
    plt.scatter(p_a[0], p_a[1], c="r", marker="*", label="mean of agent position")
    theta = np.linspace(0, 2 * np.pi, 200)
    s="a"
    x = np.cos(theta) * 22.5 + p_a[0]
    y = np.sin(theta) * 22.5 + p_a[1]
    plt.plot(x, y, color="green", linewidth=1)
    for i in range(a_num):
        plt.text(q[i, 0] + 0.2, q[i, 1] + 0.5, i,fontsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("x axis in meters", fontsize=20)
    plt.ylabel("y axis in meters", fontsize=20)
    for i in range(124):
        plt.gca().add_patch(plt.Rectangle(ob[i, 0:2], ob[i, 2], ob[i, 3], facecolor="gray"))
    plt.gca().add_patch(plt.Rectangle((0, 0), 0, 0, label="obstacle", facecolor="gray"))
    plt.scatter(500, 500, s=0, marker='o', edgecolors="green", c='none', label="$3\sigma$ area of the agent")
    plt.gca().add_patch(plt.Rectangle((1000/3,1000/3),1000/3,1000/3,facecolor="none",linestyle=":",edgecolor="r",linewidth=1,label="region of interest"))
    my_leg = plt.legend(fontsize=14)
    my_leg.legendHandles[4]._sizes = [20]
    plt.show()

    # map = np.zeros((int(20*sig_num*sig),int(20*sig_num*sig),3))
    # f = lambda x, y: 1 / (2 * np.pi * sig ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))
    # probs = 0
    # for i in range(int(20*sig_num*sig)):
    #     for j in range(int(20*sig_num*sig)):
    #         prob, err = scipy.integrate.dblquad(f, - sig_num * sig + 0.1 * i - 0.05, - sig_num * sig + 0.1 * i + 0.05,
    #                                             - sig_num * sig + 0.1 * j - 0.05, - sig_num * sig + 0.1 * j + 0.05)
    #         map[i,j,0]= - sig_num * sig + 0.1 * i - 0.05 + 500
    #         map[i,j,1]= - sig_num * sig + 0.1 * j - 0.05 + 500
    #         map[i,j,2] = prob
    #         probs += prob
    #         print(probs)
    # # Save probability distribution
    # pkl.dump(map, open("map.pkl", "wb"))

    # Load probability distribution
    map = pkl.load(open("map.pkl", "rb"))

    # # Obtain LOS map
    # LOS_map = jud_LOS(map, q, ob)
    # pkl.dump(LOS_map, open("LOS_map.pkl", "wb"))

    # Load LOS map
    LOS_map = pkl.load(open("LOS_map.pkl", "rb"))

    a_map = get_a_map(map, q)

    N_s = set()
    SPEB = [2 * sig ** 2]
    # Complete set of anchors
    U_q = set()
    # Greedy algorithm main body
    for i in range(a_num):
        U_q.add(i)

    while len(U_q.difference(N_s)):
        Upsilon= []
        Psi = []
        index = []
        for i in U_q.difference(N_s):
            Psi_temp=0
            for j in N_s|{i}:
                Psi_temp+=Lamb / (len(N_s) + 1) ** 2 * np.sum(np.sum((a_map[j, :, :] * LOS_map[j, :, :] * map[:, :, 2]),axis=1))
            Psi.append(Psi_temp)

            Upsilon_temp1=0
            Upsilon_temp2=0
            Upsilon_temp3=0
            for j in N_s|{i}:
                Upsilon_temp1+=Lamb / (len(N_s) + 1) ** 2 * np.sum(np.sum((a_map[j, :, :] * LOS_map[j, :, :] * map[:, :, 2]*np.cos(get_phi(map[:,:,0:2],q[j,:]))**2),axis=1))
                Upsilon_temp2+=Lamb / (len(N_s) + 1) ** 2 * np.sum(np.sum((a_map[j, :, :] * LOS_map[j, :, :] * map[:, :, 2]*np.sin(get_phi(map[:,:,0:2],q[j,:]))**2),axis=1))
                Upsilon_temp3+=Lamb / (len(N_s) + 1) ** 2 * np.sum(np.sum((a_map[j, :, :] * LOS_map[j, :, :] * map[:, :, 2]*np.cos(get_phi(map[:,:,0:2],q[j,:]))*np.sin(get_phi(map[:,:,0:2],q[j,:]))),axis=1))
            Upsilon.append(Upsilon_temp1*Upsilon_temp2-Upsilon_temp3**2)
            index.append(i)
        SPEB_temp = (np.array(Psi)+2*sig**(-2))/(np.array(Upsilon)+sig**(-2)*np.array(Psi)+sig**(-4))
        SPEB.append(np.min(SPEB_temp))
        N_s.add(index[np.argmin(SPEB_temp)])
        print(SPEB, N_s)

    SPEB_md = [2 * sig ** 2]
    Upsilon = []
    Psi = []
    for i in range(a_num):
        Psi_temp = 0
        Upsilon_temp1 = 0
        Upsilon_temp2 = 0
        Upsilon_temp3 = 0
        for j in range(i):
            Psi_temp += Lamb / (i + 1) ** 2 * np.sum(
                np.sum((a_map[int(dis_q[j,1]), :, :] * LOS_map[int(dis_q[j,1]), :, :] * map[:, :, 2]), axis=1))
            Upsilon_temp1 += Lamb / (i + 1) ** 2 * np.sum(
                np.sum((a_map[int(dis_q[j,1]), :, :] * LOS_map[int(dis_q[j,1]), :, :] * map[:, :, 2] * np.cos(get_phi(map[:, :, 0:2], q[int(dis_q[j,1]), :])) ** 2),
                       axis=1))
            Upsilon_temp2 += Lamb / (i + 1) ** 2 * np.sum(
                np.sum((a_map[int(dis_q[j,1]), :, :] * LOS_map[int(dis_q[j,1]), :, :] * map[:, :, 2] * np.sin(get_phi(map[:, :, 0:2], q[int(dis_q[j,1]), :])) ** 2),
                       axis=1))
            Upsilon_temp3 += Lamb / (i + 1) ** 2 * np.sum(np.sum((a_map[int(dis_q[j,1]), :, :] * LOS_map[int(dis_q[j,1]), :, :] * map[:, :,
                                                                                                             2] * np.cos(
                get_phi(map[:, :, 0:2], q[int(dis_q[j,1]), :])) * np.sin(get_phi(map[:, :, 0:2], q[int(dis_q[j,1]), :]))), axis=1))
        Psi.append(Psi_temp)
        Upsilon.append(Upsilon_temp1 * Upsilon_temp2 - Upsilon_temp3 ** 2)
        SPEB_temp = np.array((np.array(Psi) + 2 * sig ** (-2)) / (np.array(Upsilon) + sig ** (-2) * np.array(Psi) + sig ** (-4)))
        SPEB_md.append(SPEB_temp[i])
    print(SPEB_md)

    SPEB_ep = [2 * sig ** 2]
    Upsilon = []
    Psi = []
    index = []
    i = 0
    ind = 0
    while i< a_num and ind<a_num:
        Psi_temp = 0
        Upsilon_temp1 = 0
        Upsilon_temp2 = 0
        Upsilon_temp3 = 0
        while True:
            if ind >= a_num:
                break
            if LOS_map[int(dis_q[ind,1]), int(10 * sig_num * sig), int(10 * sig_num * sig)]!=0:
                index.append(ind)
                print(int(dis_q[ind,1]))
                ind+=1
                break
            else :
                ind+=1

        for j in index:
            Psi_temp += Lamb / (i + 1) ** 2 * np.sum(
                np.sum((a_map[int(dis_q[j,1]), :, :] * LOS_map[int(dis_q[j,1]), :, :] * map[:, :, 2]), axis=1))
            Upsilon_temp1 += Lamb / (i + 1) ** 2 * np.sum(
                np.sum((a_map[int(dis_q[j,1]), :, :] * LOS_map[int(dis_q[j,1]), :, :] * map[:, :, 2] * np.cos(get_phi(map[:, :, 0:2], q[int(dis_q[j,1]), :])) ** 2),
                       axis=1))
            Upsilon_temp2 += Lamb / (i + 1) ** 2 * np.sum(
                np.sum((a_map[int(dis_q[j,1]), :, :] * LOS_map[int(dis_q[j,1]), :, :] * map[:, :, 2] * np.sin(get_phi(map[:, :, 0:2], q[int(dis_q[j,1]), :])) ** 2),
                       axis=1))
            Upsilon_temp3 += Lamb / (i + 1) ** 2 * np.sum(np.sum((a_map[int(dis_q[j,1]), :, :] * LOS_map[int(dis_q[j,1]), :, :] * map[:, :,
                                                                                                             2] * np.cos(
                get_phi(map[:, :, 0:2], q[int(dis_q[j,1]), :])) * np.sin(get_phi(map[:, :, 0:2], q[int(dis_q[j,1]), :]))), axis=1))
        Psi.append(Psi_temp)
        Upsilon.append(Upsilon_temp1 * Upsilon_temp2 - Upsilon_temp3 ** 2)
        SPEB_temp = np.array((np.array(Psi) + 2 * sig ** (-2)) / (np.array(Upsilon) + sig ** (-2) * np.array(Psi) + sig ** (-4)))
        SPEB_ep.append(SPEB_temp[i])
        i+=1
    print(SPEB_ep)

    pkl.dump(SPEB, open("SPEB.pkl", "wb"))
    pkl.dump(SPEB_md, open("SPEB_md.pkl", "wb"))
    pkl.dump(SPEB_ep,open("SPEB_ep.pkl","wb"))
