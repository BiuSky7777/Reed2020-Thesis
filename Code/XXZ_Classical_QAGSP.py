from Schmidt_Decomposition import *
from AGSP import *

from XXZ_Classical_Hamiltonian import *


# H = generateHamiltonianXXZ(n,2,1)
# kls = [[i,project_random_vector(H,i)] for i in range(2,10,1)]
# print(kls)
kkls = [[2, 20], [3, 30], [4, 40], [5, 40], [6, 70], [7, 60], [8, 90], [9, 80],[10,120]]
kls = [[kkls[i][0],kkls[i][1]//2] for i in range(0,9,2)]

## Explore Kt Schmidt Coefficients dis

def Kt_Schmidt_Coeff_dis(n,limt):
    rstate = np.kron(np.random.rand(2**(n//2),1),np.random.rand(2**(n-n//2),1))
    H = generateHamiltonianXXZ(n,5,1)
    ls = []
    for t in range(0,limt+1,1):
        K = possible_AGSP(H,t)
        rstate = np.real(K.dot(rstate)/np.linalg.norm(rstate))
        sd = SD_discrete(rstate,"vector",n//2,n-n//2)
        sd_coeff = [sd[i][0] for i in range(len(sd))]
        ls.append([t,sd_coeff])
    return ls


# K_rstate_sd_ls = []
# file = open("XXZ_Classcial_AGSP_Kt_coeff_vs_n_d5_trail2even.txt","w")
# for i in range(len(kls)):
#     ls = Kt_Schmidt_Coeff_dis(kls[i][0],kls[i][1])
#     K_rstate_sd_ls.append(ls)
#     strformtc = str(ls).replace('[','{').replace(']','}').replace('e-','*10^-')
#     print("KsdCoeff"+str(kls[i][0]) + "=" + str(strformtc)+";")
#     file.write("%s\n" % ("KsdCoeff"+str(kls[i][0]) + "=" + str(strformtc)+";"))
# file.close()


## Explore Von Neumann Entropy Growth
def VNEntropyAGSP(n,limt):
    rstate = np.kron(np.random.rand(2**(n//2),1),np.random.rand(2**(n-n//2),1))
    H = generateHamiltonianXXZ(n,5,1)
    ls = []
    for t in range(0,limt+1,1):
        K = possible_AGSP(H,t)
        rstate = np.real(K.dot(rstate)/np.linalg.norm(rstate))
        sd = SD_discrete(rstate,"vector",n//2,n-n//2)
        vn_entro = sum([-sd[i][0]**2*2*np.log2(sd[i][0]) for i in range(len(sd))])
        ls.append([t,vn_entro])
    return ls

# K_rstate_entropy = []
# file = open("XXZ_Classcial_entropy_vs_n3.txt","w")
# for i in range(len(kkls)):
#     ls = VNEntropyAGSP(kkls[i][0],kkls[i][1])
#     K_rstate_entropy.append(ls)
#     strformtc = str(ls).replace('[','{').replace(']','}').replace('e-','*10^-')
#     print("KsdEntropyN"+str(kkls[i][0]) + "=" + str(strformtc)+";")
#     file.write("%s\n" % ("KsdEntropy3N"+str(kls[i][0]) + "=" + str(strformtc)+";"))
# file.close()
