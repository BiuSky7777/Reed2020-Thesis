
from AKLT_Schmidt_Decomposition import *

from AKLT_Hamiltonian import *


### AKLT Model

def AKLT_entropy_vs_n(nmax):
    entropyls = []
    for n in range(2,nmax,1):
        H = generateHamiltonianAKLT(n)
        entropyls.append([n,VNEntropy(H,n,n//2,n-n//2)])
    return entropyls
# 
# file = open("AKLT_Computation_entropy_vs_n.txt","w")
# entropy = AKLT_entropy_vs_n(9)
# strforvne = str(entropy).replace('[','{').replace(']','}').replace('e-0','*10^-')
# print("TabPlot="+strforvne+";")
# file.write("%s\n" % ("TabPlot="+strforvne+";"))
# file.close()
