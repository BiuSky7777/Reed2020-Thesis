import numpy as np

def KroneckerMultiProduct(s):
    retval = s[0]
    for i in range(1,len(s)):
        retval = np.kron(retval,s[i])
    return retval

def MultiId(M):
    Id = np.matrix([[1,0],[0,1]])
    return KroneckerMultiProduct([Id]*M)

def MakeSiteOperatorVariant(U,Num):
    ret_ls = [np.kron(U,MultiId(Num-2))]
    for i in range(2,Num-1):
        ret_ls.append(KroneckerMultiProduct([MultiId(i-1),U,MultiId(Num-i-1)]))
    ret_ls.append(np.kron(MultiId(Num-2),U))
    return ret_ls

def generateHamiltonianXXZVariant(n,q):
    Psi = 1/np.sqrt(1+q**2)*np.matrix([0,1,-q,0])
    U = np.outer(Psi,Psi)
    if n == 2:
        UTab = [U]
    else:
        UTab = MakeSiteOperatorVariant(U,n)
    H = UTab[0]
    for i in range(1,len(UTab)):
        H += UTab[i]
    return H
