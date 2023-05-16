import pickle as pkl
import numpy as np
from beam_seletction import get_beam_SPEB

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
    NT = 36
    beam_SPEB_Nb = []
    F = np.zeros((NT, NT), dtype=complex)
    for i in range(NT):
        for j in range(int(NT / 2)):
            F[i, j] = np.exp(-1j * np.pi * i * (4 * j / NT - 1))

    for i in range(0,18):
        beam_SPEB = get_beam_SPEB(q[11], 0, F[:, i], LOS_map[0, 11], all_map, a_map[0, 11],
                                  range(int(K/2)))
        beam_SPEB_Nb.append(beam_SPEB)
        print(i, beam_SPEB)
    pkl.dump(beam_SPEB_Nb,open("beam_SPEB_Nb.pkl","wb"))