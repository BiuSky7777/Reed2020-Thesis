from Schmidt_Decomposition import *

from XXZ_Schmidt_Decomposition import *

from XXZ_Classical_Hamiltonian import *
from XXZ_Variant_Hamiltonian import *

### XXZ Model

def entropy_vs_n(delta,J,nmax):
    entropyls = []
    for n in range(2,nmax,1):
        H = generateHamiltonianXXZGr(n,delta,J)
        entropyls.append([n,VNEntropy(H,n,n//2,n-n//2)])
    return entropyls

# delta_ls = [i for i in range(0,20)] + [20,35,50,65,80,99] +  [i for i in range(200,1000,100)] + [1000,10000,50000,100000,500000,1000000]
# file = open("XXZ_Computation_entropy_vs_n_gr.txt","w")
# for d in delta_ls:
#     entropy = entropy_vs_n(d,1,15)
#     strforvne = str(entropy).replace('[','{').replace(']','}').replace('e-','*10^-')
#     print("TabPlotD"+str(d)+"="+strforvne+";")
#     file.write("%s\n" % ("TabPlotD" + str(d) + " = "+strforvne+";"))
# file.close()



def entropy_vs_p(n,delta,J):
    entropyls = []
    for dA in range(1,np.int_(n/2)+1):
        dB = n-dA
        H = generateHamiltonianXXZGr(n,delta,J)
        entropyls.append([dA,VNEntropy(H,n,dA,dB)])
    return entropyls

# n_ls = [i for i in range(2,15)]
# delta_ls = [2,5,10,49]
# file = open("XXZ_Computation_entropy_vs_p.txt","w")
# for d in delta_ls:
#     for n in n_ls:
#         entropy = entropy_vs_p(n,d,1)
#         strforvne = str(entropy).replace('[','{').replace(']','}').replace('e-','*10^-')
#         print("TabPlotD"+str(d)+"N"+str(n)+"="+strforvne+";")
#         file.write("%s\n" % ("TabPlotD"+str(d)+"N"+str(n)+"="+strforvne+";"))
# file.close()

def sdcoeff_vs_n(delta,J,nmax):
    sd_coeff_ls = []
    for n in range(2,nmax):
        H = generateHamiltonianXXZGr(n,delta,J)
        sd_coeff_ls.append([n,SDCoeff(H,n,n//2,n-n//2)])
    return sd_coeff_ls

# delta_ls = [i for i in range(0,20)] + [20,35,50,65,80,99]
# file = open("XXZ_Computation_sd_coeff_vs_n_gr2.txt","w")
# for d in delta_ls:
#     sdcoeff = sdcoeff_vs_n(d,1,14)
#     strforsdcoeff = str(sdcoeff).replace('[','{').replace(']','}').replace('e-','*10^-')
#     print("SDTabPlotD"+str(d)+"="+strforsdcoeff+";")
#     file.write("%s\n" % ("SDTabPlotD"+str(d)+"="+strforsdcoeff+";"))
# file.close()


def sdcoeff_vs_n_predict(delta,J,n):
    H = generateHamiltonianXXZGr(n,delta,J)
    return [n,SDCoeff(H,n,n//2,n-n//2)]

# delta_ls = [i for i in range(0,20)] + [20,35,50,65,80,99]
# file = open("XXZ_Computation_sd_coeff_vs_n_predict.txt","w")
# for d in delta_ls:
#     sdcoeff = sdcoeff_vs_n_predict(d,1,14)
#     strforsdcoeff = str(sdcoeff).replace('[','{').replace(']','}').replace('e-','*10^-')
#     print("SDTabPlotD"+str(d)+"N14="+strforsdcoeff+";")
#     file.write("%s\n" % ("SDTabPlotD"+str(d)+"N14="+strforsdcoeff+";"))
# file.close()





### XXZ Variant Model

def Variant_entropy_vs_n(q,nmax):
    entropyls = []
    for n in range(2,nmax,1):
        H = generateHamiltonianXXZVariant(n,q)
        entropyls.append([n,VNEntropy(H,n,n//2,n-n//2)])
    return entropyls

# q_ls = [i for i in range(0,20)] + [20,35,50,65,80,99]
# file = open("XXZ_Computation_Variant_entropy_vs_n.txt","w")
# for q in q_ls:
#     entropy = Variant_entropy_vs_n(q,13)
#     strforvne = str(entropy).replace('[','{').replace(']','}').replace('e-0','*10^-')
#     print("TabPlotQ"+str(q)+"="+strforvne+";")
#     file.write("%s\n" % ("TabPlotQ"+str(q)+"="+strforvne+";"))
# file.close()
