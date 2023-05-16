import pickle as pkl
import numpy as np
from beam_seletction import get_phi,get_Delta
from matplotlib import pyplot as plt

# system parameter
sig = 7.5
sig_num = 3
c = 3e8
B = 1e7
K = 32
M = 16
a_num = 36
N0 = 1e-20
Pt = 0.01
Ts = 1/B

def get_beam_EFIM(q,p_index,f,LoS_map,all_map,a_map,list_K):
    NT = len(f)
    f=f.reshape((-1,1))
    EFIM = np.zeros((2,2))
    for i in range(LoS_map.shape[0]):
        for j in range(LoS_map.shape[1]):
            theta=get_phi([all_map[p_index,i,j,0],all_map[p_index,i,j,1]],q)
            dis = np.sqrt((all_map[p_index,i,j,0]-q[0])**2+(all_map[p_index,i,j,1]-q[1])**2)
            EFIM_tau=np.dot(np.array([np.cos(theta),np.sin(theta)]).reshape(-1,1)/c,np.array([np.cos(theta),np.sin(theta)]).reshape(1,-1)/c)
            EFIM_theta=np.dot(np.array([-np.sin(theta),np.cos(theta)]).reshape(-1,1)/dis,np.array([-np.sin(theta),np.cos(theta)]).reshape(1,-1)/dis)
            a=np.zeros((NT,1),dtype=complex)
            D=np.zeros((NT,NT),dtype=complex)
            for m in range(NT):
                a[m, 0] = np.exp(-1j * m * np.pi * np.sin(theta))
                D[m, m] = -1j * np.pi * np.cos(theta) * m
            A = np.dot(a, a.conjugate().T) / NT
            J_tau = 0
            for k in list_K:
                J_tau += get_Delta(k)**2
            J_tau = 2*NT*Pt/(B/K*N0)*a_map[i,j]*M*J_tau*np.dot(np.dot(f.conjugate().T,A),f).real
            J_theta = 2*NT*Pt/(B/K*N0)*a_map[i,j]*M*len(list_K)*np.dot(np.dot(np.dot(f.conjugate().T,D.conjugate().T),A),np.dot(D,f)).real
            J_alpha = 2*NT*Pt/(B/K*N0)*M*len(list_K)*np.dot(np.dot(f.conjugate().T,A),f).real
            J_t_a = 4*(NT*Pt/(B/K*N0)*M*len(list_K))**2*a_map[i,j]\
                    *np.dot(np.dot(np.dot(f.conjugate().T,D.conjugate().T),np.dot(np.dot(A,f),np.dot(f.conjugate().T,A.conjugate().T))),np.dot(D,f)).real
            if J_alpha != 0:
                J_theta_bar=J_theta-J_t_a/J_alpha
                EFIM_p = J_theta_bar*EFIM_theta+J_tau*EFIM_tau
                EFIM+=LoS_map[i,j]*all_map[p_index,i,j,2]*EFIM_p
            else:
                EFIM += np.zeros((2,2))
    EFIM[0,0]+=sig**-2
    EFIM[1,1]+=sig**-2
    return EFIM

def get_beamforming_2dis(p,q,p_index,q_index,LoS_map,all_map,a_map,NT,list_K):
    EFIM = np.zeros((2,2))
    for j in range(len(q_index)):
        phi=get_phi(p,q[j])
        i = int(np.floor((np.sin(phi)+1)*NT/4))

        EFIM += get_beam_EFIM(q[j],p_index,F[:,i],LoS_map[p_index,q_index[j]],all_map,a_map[p_index,q_index[j]],list_K)
        print("第",j,"地址的波束号为",i)

    inv_EFIM = np.linalg.inv(EFIM)
    return inv_EFIM[0,0]+inv_EFIM[1,1]

def get_anchor_index_2dis(q,p,p_index):
    anchor_index=[]
    dis = np.zeros((len(p_index),q.shape[0]))
    for i in range(len(p_index)):
        for j in range(q.shape[0]):
            dis[i,j]=(p[p_index[i],0]-q[j,0])**2+(p[p_index[i],1]-q[j,1])**2
    for index in range(len(p_index)):
        if index == 0:
            anchor_index.append([np.argsort(dis[index])[0],np.argsort(dis[index])[1]])
        else:
            order = 0
            anchor_second_index = []
            while len(anchor_second_index)!=2:
                if np.argsort(dis[index])[order] in np.ravel(anchor_index):
                    order +=1
                else:
                    anchor_second_index.append(np.argsort(dis[index])[-order])
                    order += 1
            anchor_index.append(anchor_second_index)
    return anchor_index

if __name__ == "__main__":
    ob = pkl.load(open("ob.pkl", "rb"))
    q = pkl.load(open("q.pkl", "rb"))
    p = pkl.load(open("p.pkl", "rb"))

    # Load probability distribution
    map = pkl.load(open("map.pkl", "rb"))

    all_map = pkl.load(open("all_map.pkl", "rb"))
    # Load LOS map
    LOS_map = pkl.load(open("all_LOS_map.pkl", "rb"))

    a_map = pkl.load(open("a_map.pkl", "rb"))

    LoS_prob = pkl.load(open("LoS_prob.pkl", "rb"))

    beam_SPEB_NT = np.zeros((1,2))

    for NT in range(80,112,16):
        F=np.zeros((NT,NT),dtype=complex)
        for i in range(NT):
            for j in range(int(NT/2)):
                F[i,j]=np.exp(-1j*np.pi*i*(4*j/NT-1))
        beam_SPEB_dis = 0

        beam_SPEB_0_dis = get_beamforming_2dis(p[0], [q[12],q[14]], 0, [12,14], LOS_map, all_map, a_map, NT,
                                      range(int(K / 4)))
        for i in range(1,10):
            p_list=[0]+[i]
            # #多锚点覆盖方法
            anchor_index=get_anchor_index_2dis(q,p,p_list)
            beam_SPEB_1_dis = get_beamforming_2dis(p[i], [q[anchor_index[1][0]],q[anchor_index[1][1]]], p_list[1], anchor_index[1], LOS_map, all_map,
                                              a_map, NT, range(int(K / 2), int(3*K / 4)))
            beam_SPEB_dis = beam_SPEB_0_dis + beam_SPEB_1_dis
            print("SPEB for all user of 2 distance method:",beam_SPEB_dis)
        beam_SPEB_NT[0,int(NT/16)-5]=beam_SPEB_dis/9
    pkl.dump(beam_SPEB_NT,open("beam_SPEB_2dis_NT2.pkl","wb"))