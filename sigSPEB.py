import pickle as pkl
import scipy.integrate
import numpy as np
import time

# system parameter
sig = 5
sig_num = 3
N_0 = 1e-20
c = 3e8
f = 4.5e9
Es = 1e-11
a_num = 36
B_s=100e6

def Lamb(B_s):
    return 8 * np.pi ** 2 * B_s ** 2 * Es / (3 * c ** 2 * N_0)

def jud_LOS(map, q, ob):
    LOS_map = np.ones((10,10,a_num, int(20 * sig_num * sig), int(20 * sig_num * sig)))
    t0=time.time()
    for l in range(10):
        for k in range(10):
            for n in range(a_num):
                t=time.time()
                for i in range(int(20 * sig_num * sig)):
                    for j in range(int(20 * sig_num * sig)):
                        for m in range(ob.shape[0]):
                            if map[k,l,i, j, 0] < ob[m, 0] < q[n, 0] or q[n, 0] < ob[m, 0] < map[k,l,i, j, 0]:
                                if ob[m, 1] < (map[k,l,i, j, 1] - q[n, 1]) / (map[k,l,i, j, 0] - q[n, 0]) * (ob[m, 0] - map[k,l,i, j, 0]) \
                                        + map[k,l,i, j, 1] < ob[m, 1] + ob[m, 3]:
                                    LOS_map[k,l,n, i, j] = 0
                                    break
                            if map[k,l,i, j, 0] < ob[m, 0] + ob[m, 2] < q[n, 0] or q[n, 0] < ob[m, 0] + ob[m, 2] < map[k,l,
                                i, j, 0]:
                                if ob[m, 1] < (map[k,l,i, j, 1] - q[n, 1]) / (map[k,l,i, j, 0] - q[n, 0]) * (
                                        ob[m, 0] + ob[m, 2] - map[k,l,i, j, 0]) \
                                        + map[k,l,i, j, 1] < ob[m, 1] + ob[m, 3]:
                                    LOS_map[k,l,n, i, j] = 0
                                    break
                            if map[k,l,i, j, 1] < ob[m, 1] < q[n, 1] or q[n, 1] < ob[m, 1] < map[k,l,i, j, 1]:
                                if ob[m, 0] < (map[k,l,i, j, 0] - q[n, 0]) / (map[k,l,i, j, 1] - q[n, 1]) * (ob[m, 1] - map[k,l,i, j, 1]) \
                                        + map[k,l,i, j, 0] < ob[m, 0] + ob[m, 2]:
                                    LOS_map[k,l,n, i, j] = 0
                                    break
                            if map[k,l,i, j, 1] < ob[m, 1] + ob[m, 3] < q[n, 1] or q[n, 1] < ob[m, 1] + ob[m, 3] < map[
                                k,l,i, j, 1]:
                                if ob[m, 0] < (map[k,l,i, j, 0] - q[n, 0]) / (map[k,l,i, j, 1] - q[n, 1]) * (
                                        ob[m, 1] + ob[m, 3] - map[k,l,i, j, 1]) \
                                        + map[k,l,i, j, 0] < ob[m, 0] + ob[m, 2]:
                                    LOS_map[k,l,n, i, j] = 0
                                    break
                print(l,k,n,"轮计算用时",time.time()-t,"累计用时",np.floor((time.time()-t0)/3600),"时",
                      np.floor(((time.time()-t0)-3600*np.floor((time.time()-t0)/3600))/60),"分",np.mod(time.time()-t0,60),"秒")
                print("预计剩余时间",np.floor(((time.time()-t0)/(360*l+36*k+n+1)*(3599-360*l-36*k-n))/3600),"时",
                      np.floor((((time.time()-t0)/(360*l+36*k+n+1)*(3599-360*l-36*k-n))-3600*np.floor(((time.time()-t0)/(360*l+36*k+n+1)*(3599-360*l-36*k-n))/3600))/60),"分",np.mod((time.time()-t0)/(360*l+36*k+n+1)*(3599-360*l-36*k-n),60),"秒")
    return LOS_map

def get_a_map(map, q):
    a = np.zeros((10,10,a_num, int(20 * sig_num * sig), int(20 * sig_num * sig)))
    for l in range(10):
        for k in range(10):
            for i in range(a_num):
                a[k,l,i, :] = (c / (4 * np.pi * f)) ** 2 / ((map[k,l,:, :, 0] - q[i, 0]) ** 2 + (map[k,l,:, :, 1] - q[i, 1]) ** 2)
    return a

def get_phi(p, q):
    return np.arctan((p[:, :, 1] - q[1]) / (p[:, :, 0] - q[0]))


if __name__ == "__main__":
    # set the size and position of the obstacle
    ob = pkl.load(open("ob.pkl", "rb"))
    q = pkl.load(open("q.pkl", "rb"))
    p = pkl.load(open("p.pkl", "rb"))
    p_a = [500, 500]

    # map = np.zeros((10,int(20 * sig_num * sig), int(20 * sig_num * sig), 3))
    # func = lambda x, y: 1 / (2 * np.pi * ((n + 1) * sig) ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * ((n + 1) * sig) ** 2))
    # for n in range(10):
    #     probs = 0
    #     for i in range(int(20*sig_num*sig)):
    #         for j in range(int(20*sig_num*sig)):
    #             prob, err = scipy.integrate.dblquad(func, (n+1)* (- sig_num * sig + 0.1 * i- 0.05),(n+1)*( - sig_num * sig + 0.1 * i + 0.05),
    #                                                 (n+1)*(- sig_num * sig + 0.1 * j - 0.05), (n+1)*(- sig_num * sig + 0.1 * j + 0.05))
    #             map[n,i,j,0]= (n+1)*(- sig_num * sig + 0.1 * i - 0.05) + 500
    #             map[n,i,j,1]= (n+1)*(- sig_num * sig + 0.1 * j - 0.05) + 500
    #             map[n,i,j,2] = prob
    #             probs += prob
    #             print(probs)
    #
    # # Save probability distribution
    # pkl.dump(map, open("sig_map.pkl", "wb"))

    # Load probability distribution
    map = pkl.load(open("sig_map.pkl", "rb"))

    all_map=np.zeros((10,10,int(20*sig_num*sig),int(20*sig_num*sig),3))
    for i in range(10):
        for n in range(10):
            all_map[i,n,:,:,0]=map[n,:,:,0]-500+p[i,0]
            all_map[i,n,:, :, 1] = map[n,:, :, 1] - 500 + p[i, 1]
            all_map[i,n,:,:,2]=map[n,:,:,2]

    # # # Obtain LOS map
    # LOS_map = jud_LOS(all_map, q, ob)
    # pkl.dump(LOS_map, open("all_sig_LOS_map.pkl", "wb"))

    # Load LOS map
    LOS_map = pkl.load(open("all_sig_LOS_map.pkl", "rb"))

    a_map = get_a_map(all_map, q)

    avg_SPEB=np.zeros((10,10))
    avg_SPEB_md=np.zeros((11,10,10))
    avg_allSPEB=np.zeros((10,10))
    avg_SPEB_ep=np.zeros((10,10))

    for n in range(10):
        for k in range(10):
            t=time.time()
            print(n,k)
            N_s = set()
            SPEB = [2 * ((n+1)*sig) ** 2]
            # Complete set of anchors
            U_q = set()
            # Greedy algorithm main body
            for i in range(a_num):
                U_q.add(i)

            while len(U_q.difference(N_s)):
                Upsilon = []
                Psi = []
                index = []
                for i in U_q.difference(N_s):
                    Psi_temp = 0
                    for j in N_s | {i}:
                        Psi_temp += Lamb(B_s) / (len(N_s) + 1) ** 2 * np.sum(
                            np.sum((a_map[k,n,j, :, :] * LOS_map[k,n,j, :, :] * all_map[k,n,:, :, 2]), axis=1))
                    Psi.append(Psi_temp)

                    Upsilon_temp1 = 0
                    Upsilon_temp2 = 0
                    Upsilon_temp3 = 0
                    for j in N_s | {i}:
                        Upsilon_temp1 += Lamb(B_s) / (len(N_s) + 1) ** 2 * np.sum(np.sum((a_map[k,n,j, :, :] * LOS_map[k,n,j, :,
                                                                                                      :] * all_map[k,n,:, :,
                                                                                                           2] * np.cos(
                            get_phi(all_map[k,n,:, :, 0:2], q[j, :])) ** 2), axis=1))
                        Upsilon_temp2 += Lamb(B_s) / (len(N_s) + 1) ** 2 * np.sum(np.sum((a_map[k,n,j, :, :] * LOS_map[k,n,j, :,
                                                                                                      :] * all_map[k,n,:, :,
                                                                                                           2] * np.sin(
                            get_phi(all_map[k,n,:, :, 0:2], q[j, :])) ** 2), axis=1))
                        Upsilon_temp3 += Lamb(B_s) / (len(N_s) + 1) ** 2 * np.sum(np.sum((a_map[k,n,j, :, :] * LOS_map[k,n,j, :,
                                                                                                      :] * all_map[k,n,:, :,
                                                                                                           2] * np.cos(
                            get_phi(all_map[k,n,:, :, 0:2], q[j, :])) * np.sin(get_phi(all_map[k,n,:, :, 0:2], q[j, :]))), axis=1))
                    Upsilon.append(Upsilon_temp1 * Upsilon_temp2 - Upsilon_temp3 ** 2)
                    index.append(i)
                SPEB_temp = (np.array(Psi) + 2 * ((n+1)*sig) ** (-2)) / (
                            np.array(Upsilon) + ((n+1)*sig) ** (-2) * np.array(Psi) + ((n+1)*sig) ** (-4))
                if np.min(SPEB_temp)>np.min(SPEB):
                    break
                SPEB.append(np.min(SPEB_temp))
                N_s.add(index[np.argmin(SPEB_temp)])
            avg_SPEB[n, k] = np.min(SPEB)

            # Distance of anchors
            dis_q = np.zeros((a_num, 2))
            for i in range(a_num):
                dis_q[i, 1] = i
                dis_q[i, 0] = np.sqrt((q[i, 0]-p[k,0]) ** 2 + (q[i, 1]-p[k,1]) ** 2)
            dis_q = dis_q[np.lexsort(dis_q[:, ::-1].T)]

            SPEB_md = [2 * ((n+1)*sig) ** 2]
            Upsilon = []
            Psi = []
            index = []
            fix_num=10
            for i in range(fix_num):
                Psi_temp = 0
                Upsilon_temp1 = 0
                Upsilon_temp2 = 0
                Upsilon_temp3 = 0
                for j in range(i):
                    Psi_temp += Lamb(B_s) / (i + 1) ** 2 * np.sum(
                        np.sum((a_map[k,n,int(dis_q[j, 1]), :, :] * LOS_map[k,n,int(dis_q[j, 1]), :, :] * all_map[k,n,:, :, 2]),
                               axis=1))
                    Upsilon_temp1 += Lamb(B_s) / (i + 1) ** 2 * np.sum(
                        np.sum((a_map[k,n,int(dis_q[j, 1]), :, :] * LOS_map[k,n,int(dis_q[j, 1]), :, :] * all_map[k,n,:, :, 2] * np.cos(
                            get_phi(all_map[k,n,:, :, 0:2], q[int(dis_q[j, 1]), :])) ** 2),
                               axis=1))
                    Upsilon_temp2 += Lamb(B_s) / (i + 1) ** 2 * np.sum(
                        np.sum((a_map[k,n,int(dis_q[j, 1]), :, :] * LOS_map[k,n,int(dis_q[j, 1]), :, :] * all_map[k,n,:, :, 2] * np.sin(
                            get_phi(all_map[k,n,:, :, 0:2], q[int(dis_q[j, 1]), :])) ** 2),
                               axis=1))
                    Upsilon_temp3 += Lamb(B_s) / (i + 1) ** 2 * np.sum(
                        np.sum((a_map[k,n,int(dis_q[j, 1]), :, :] * LOS_map[k,n,int(dis_q[j, 1]), :, :] * all_map[k,n,:, :,
                                                                                                  2] * np.cos(
                            get_phi(all_map[k,n,:, :, 0:2], q[int(dis_q[j, 1]), :])) * np.sin(
                            get_phi(all_map[k,n,:, :, 0:2], q[int(dis_q[j, 1]), :]))), axis=1))
                Psi.append(Psi_temp)
                Upsilon.append(Upsilon_temp1 * Upsilon_temp2 - Upsilon_temp3 ** 2)
                SPEB_temp = np.array(
                    (np.array(Psi) + 2 * ((n+1)*sig) ** (-2)) / (np.array(Upsilon) + ((n+1)*sig) ** (-2) * np.array(Psi) + ((n+1)*sig) ** (-4)))
                SPEB_md.append(SPEB_temp[i])
            avg_SPEB_md[0,n,k]=2 * ((n+1)*sig) ** 2
            for i in range(fix_num):
                    avg_SPEB_md[i+1,n,k]=SPEB_md[i]

            Psi = 0
            Upsilon_temp1 = 0
            Upsilon_temp2 = 0
            Upsilon_temp3 = 0
            for j in range(a_num):
                Psi += Lamb(B_s) / (a_num) ** 2 * np.sum(np.sum((a_map[k,n,j, :, :] * LOS_map[k,n,j, :, :] * all_map[k,n,:, :, 2]), axis=1))
                Upsilon_temp1 += Lamb(B_s) / (a_num) ** 2 * np.sum(np.sum((a_map[k,n,j, :, :] * LOS_map[k,n,j, :,
                                                                                                  :] * all_map[k,n,:, :,
                                                                                                       2] * np.cos(
                        get_phi(all_map[k,n,:, :, 0:2], q[j, :])) ** 2), axis=1))
                Upsilon_temp2 += Lamb(B_s) / (a_num) ** 2 * np.sum(np.sum((a_map[k,n,j, :, :] * LOS_map[k,n,j, :,
                                                                                                  :] * all_map[k,n,:, :,
                                                                                                       2] * np.sin(
                        get_phi(all_map[k,n,:, :, 0:2], q[j, :])) ** 2), axis=1))
                Upsilon_temp3 += Lamb(B_s) / (a_num) ** 2 * np.sum(np.sum((a_map[k,n,j, :, :] * LOS_map[k,n,j, :,
                                                                                                  :] * all_map[k,n,:, :,
                                                                                                       2] * np.cos(
                        get_phi(all_map[k,n,:, :, 0:2], q[j, :])) * np.sin(get_phi(all_map[k,n,:, :, 0:2], q[j, :]))), axis=1))
            Upsilon=(Upsilon_temp1 * Upsilon_temp2 - Upsilon_temp3 ** 2)
            avg_allSPEB[n,k]=(Psi + 2 * ((n+1)*sig) ** (-2)) / (Upsilon + ((n+1)*sig) ** (-2) * Psi + ((n+1)*sig) ** (-4))

            SPEB_ep = [2 * ((n+1)*sig) ** 2]
            Upsilon = []
            Psi = []
            index = []
            i = 0
            ind = 0
            while i < a_num and ind < a_num and len(index)<3:
                Psi_temp = 0
                Upsilon_temp1 = 0
                Upsilon_temp2 = 0
                Upsilon_temp3 = 0
                while True:
                    if ind >= a_num:
                        break
                    if LOS_map[k,n,int(dis_q[ind, 1]), int(10 * sig_num * sig), int(10 * sig_num * sig)] != 0:
                        index.append(ind)
                        ind += 1
                        break
                    else:
                        ind += 1

                for j in index:
                    Psi_temp += Lamb(B_s) / (i + 1) ** 2 * np.sum(
                        np.sum((a_map[k,n,int(dis_q[j, 1]), :, :] * LOS_map[k,n,int(dis_q[j, 1]), :, :] * all_map[k,n,:, :, 2]),
                               axis=1))
                    Upsilon_temp1 += Lamb(B_s) / (i + 1) ** 2 * np.sum(
                        np.sum((a_map[k,n,int(dis_q[j, 1]), :, :] * LOS_map[k,n,int(dis_q[j, 1]), :, :] * all_map[k,n,:, :, 2] * np.cos(
                            get_phi(all_map[k,n,:, :, 0:2], q[int(dis_q[j, 1]), :])) ** 2),
                               axis=1))
                    Upsilon_temp2 += Lamb(B_s) / (i + 1) ** 2 * np.sum(
                        np.sum((a_map[k,n,int(dis_q[j, 1]), :, :] * LOS_map[k,n,int(dis_q[j, 1]), :, :] * all_map[k,n,:, :, 2] * np.sin(
                            get_phi(all_map[k,n,:, :, 0:2], q[int(dis_q[j, 1]), :])) ** 2),
                               axis=1))
                    Upsilon_temp3 += Lamb(B_s) / (i + 1) ** 2 * np.sum(
                        np.sum((a_map[k,n,int(dis_q[j, 1]), :, :] * LOS_map[k,n,int(dis_q[j, 1]), :, :] * all_map[k,n,:, :,2] * np.cos(
                            get_phi(all_map[k,n,:, :, 0:2], q[int(dis_q[j, 1]), :])) * np.sin(
                            get_phi(all_map[k,n,:, :, 0:2], q[int(dis_q[j, 1]), :]))), axis=1))
                Psi.append(Psi_temp)
                Upsilon.append(Upsilon_temp1 * Upsilon_temp2 - Upsilon_temp3 ** 2)
                SPEB_temp = np.array(
                    (np.array(Psi) + 2 * ((n+1)*sig) ** (-2)) / (np.array(Upsilon) + ((n+1)*sig) ** (-2) * np.array(Psi) + ((n+1)*sig) ** (-4)))
                SPEB_ep.append(SPEB_temp[i])
                i += 1
            avg_SPEB_ep[n,k]=np.min(SPEB_ep)

    print(np.sum(avg_SPEB,axis=1)/10)
    for i in range(11):
        print(np.sum(avg_SPEB_md[i], axis=1) / 10)
    print(np.sum(avg_allSPEB,axis=1)/10)
    print(np.sum(avg_SPEB_ep,axis=1)/10)

    pkl.dump(avg_SPEB, open("avg_sigSPEB.pkl", "wb"))
    pkl.dump(avg_SPEB_md, open("avg_sigSPEB_md.pkl", "wb"))
    pkl.dump(avg_allSPEB, open("avg_sigallSPEB.pkl", "wb"))
    pkl.dump(avg_SPEB_ep, open("avg_sigSPEB_ep.pkl", "wb"))