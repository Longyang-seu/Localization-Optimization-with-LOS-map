import pickle as pkl
import numpy as np
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

def get_phi(p, q):
    return np.arctan((p[1] - q[1]) / (p[0] - q[0]))

def get_LoS_prob(p,q,map,LoS_map):
    LoS_prob=np.zeros((p.shape[0],q.shape[0]))
    for n in range(p.shape[0]):
        for k in range(q.shape[0]):
            for i in range(map.shape[0]):
                for j in range(map.shape[1]):
                    LoS_prob[n,k]+=map[i,j,2]*LoS_map[n,k,i,j]
    return LoS_prob

def get_anchor_index(prob,p_index):
    anchor_index=[]
    for index in p_index:
        anchor_index.append(prob[index].argmax())
    if anchor_index[0]==anchor_index[1]:
        new_prob=prob.copy()
        anchor_index_sec = []
        prob_sec=[]
        for index in p_index:
            new_prob[index,anchor_index[0]]=0
            prob_sec.append(max(new_prob[index]))
            anchor_index_sec.append(new_prob[index].argmax())
        if prob_sec[0]>prob_sec[1]:
            anchor_index[0]=anchor_index_sec[0]
        else:
            anchor_index[1]=anchor_index_sec[1]
    return anchor_index

def get_anchor_index_dis(q,p,p_index):
    anchor_index=[]
    dis = np.zeros((len(p_index),q.shape[0]))
    for i in range(len(p_index)):
        for j in range(q.shape[0]):
            dis[i,j]=(p[p_index[i],0]-q[j,0])**2+(p[p_index[i],1]-q[j,1])**2
    for index in range(len(p_index)):
        anchor_index.append(dis[index].argmin())
    if anchor_index[0]==anchor_index[1]:
        new_dis=dis.copy()
        anchor_index_sec = []
        prob_sec=[]
        for index in range(len(p_index)):
            new_dis[index,anchor_index[0]]=1e6
            prob_sec.append(min(new_dis[index]))
            anchor_index_sec.append(new_dis[index].argmin())
        if prob_sec[0]<prob_sec[1]:
            anchor_index[0]=anchor_index_sec[0]
        else:
            anchor_index[1]=anchor_index_sec[1]
    return anchor_index

def get_Delta(k):
    return 2*np.pi*k/(K*Ts)

def get_beam_SPEB(q,p_index,f,LoS_map,all_map,a_map,list_K):
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
    inv_EFIM = np.linalg.inv(EFIM)
    return inv_EFIM[0,0]+inv_EFIM[1,1]

def get_beamforming(p,q,p_index,q_index,LoS_map,all_map,a_map,NT,list_K):
    phi=[0]*5
    phi[0]=get_phi(p,q)
    phi[1]=get_phi(np.add(p,[22.5,-22.5]),q)
    phi[2]=get_phi(np.add(p,[-22.5,22.5]),q)
    phi[3]=get_phi(np.add(p,[-22.5,-22.5]),q)
    phi[4]=get_phi(np.add(p,[22.5,22.5]),q)
    phi_min = min(phi)
    phi_max = max(phi)
    print("可能的波束号范围为",int(np.floor((np.sin(phi_min)+1)*NT/4)),int(np.ceil((np.sin(phi_max)+1)*NT/4)))

    best_beamforming = 0
    min_SPEB=2*sig**2

    for i in range(int(np.floor((np.sin(phi_min)+1)*NT/4)),int(np.ceil((np.sin(phi_max)+1)*NT/4))+1):
        beam_SPEB = get_beam_SPEB(q,p_index,F[:,i],LoS_map[p_index,q_index],all_map,a_map[p_index,q_index],list_K)
        print(p_index,i,beam_SPEB)
        if beam_SPEB < min_SPEB:
            best_beamforming = i
            min_SPEB = beam_SPEB
    return min_SPEB

def get_beamforming_dis(p,q,p_index,q_index,LoS_map,all_map,a_map,NT,list_K):
    phi=get_phi(p,q)
    i = int(np.floor((np.sin(phi)+1)*NT/4))

    beam_SPEB = get_beam_SPEB(q,p_index,F[:,i],LoS_map[p_index,q_index],all_map,a_map[p_index,q_index],list_K)
    print("地址的波束号为",i,beam_SPEB)
    return beam_SPEB

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

    beam_SPEB_NT = np.zeros((2,4))

    for NT in range(16,80,16):
        F=np.zeros((NT,NT),dtype=complex)
        for i in range(NT):
            for j in range(int(NT/2)):
                F[i,j]=np.exp(-1j*np.pi*i*(4*j/NT-1))
        beam_SPEB_map = 0
        beam_SPEB_dis = 0

        beam_SPEB_0_map = get_beamforming(p[0], q[19], 0, 19, LOS_map, all_map, a_map, NT,
                                      range(int(K / 2)))
        beam_SPEB_0_dis = get_beamforming_dis(p[0], q[12], 0, 12, LOS_map, all_map, a_map, NT,
                                      range(int(K / 2)))
        for i in range(1,10):
            p_list=[0]+[i]
            # #MAP方法
            anchor_index=get_anchor_index(LoS_prob,p_list)
            beam_SPEB_1 = get_beamforming(p[i], q[anchor_index[1]], p_list[1], anchor_index[1], LOS_map, all_map, a_map,
                                          NT, range(int(K / 2), K))
            if anchor_index[0]!=19:
                beam_SPEB_map = beam_SPEB_1 + get_beamforming(p[0], q[anchor_index[0]], 0, anchor_index[0], LOS_map, all_map, a_map, NT,range(int(K / 2)))
            else:
                beam_SPEB_map += beam_SPEB_0_map + beam_SPEB_1
            print("SPEB for all user of map method:",beam_SPEB_map)
            # #距离方法
            anchor_index=get_anchor_index_dis(q,p,p_list)
            beam_SPEB_1 = get_beamforming_dis(p[i], q[anchor_index[1]], p_list[1], anchor_index[1], LOS_map, all_map,
                                              a_map, NT, range(int(K / 2), K))
            if anchor_index[0]!=12:
                beam_SPEB_dis = beam_SPEB_1 + get_beamforming_dis(p[0], q[anchor_index[0]], 0, anchor_index[0], LOS_map, all_map, a_map, NT,range(int(K / 2)))
            else:
                beam_SPEB_dis += beam_SPEB_0_dis + beam_SPEB_1
            print("SPEB for all user of distance method:",beam_SPEB_dis)
            # #多锚点覆盖方法
            anchor_index=get_anchor_index_dis(q,p,p_list)
            beam_SPEB_1 = get_beamforming_dis(p[i], q[anchor_index[1]], p_list[1], anchor_index[1], LOS_map, all_map,
                                              a_map, NT, range(int(K / 2), K))
            if anchor_index[0]!=12:
                beam_SPEB_dis = beam_SPEB_1 + get_beamforming_dis(p[0], q[anchor_index[0]], 0, anchor_index[0], LOS_map, all_map, a_map, NT,range(int(K / 2)))
            else:
                beam_SPEB_dis += beam_SPEB_0_dis + beam_SPEB_1
            print("SPEB for all user of 2 distance method:",beam_SPEB_dis)
        beam_SPEB_NT[0,int(NT/16)-1]=beam_SPEB_map/9
        beam_SPEB_NT[1,int(NT/16)-1]=beam_SPEB_dis/9
    pkl.dump(beam_SPEB_NT,open("beam_SPEB_NT.pkl","wb"))