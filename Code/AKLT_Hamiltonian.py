import numpy as np

def KroneckerMultiProduct(s):
    retval = s[0]
    for i in range(1,len(s)):
        retval = np.kron(retval,s[i])
    return retval

def MultiId(M):
    Id = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    return KroneckerMultiProduct([Id]*M)

def MakeSiteOperator(U,Num):
    ret_ls = [np.kron(U,MultiId(Num-1))]
    for i in range(2,Num):
        ret_ls.append(KroneckerMultiProduct([MultiId(i-1),U,MultiId(Num-i)]))
    ret_ls.append(np.kron(MultiId(Num-1),U))
    return ret_ls

def generateHamiltonianAKLT(n):
    X = 1/(2**0.5)*np.matrix([[0,1,0],[1,0,1],[0,1,0]])
    Y = 1/(2**0.5)*np.matrix([[0,-1j,0],[1j,0,-1j],[0,1j,0]])
    Z = np.matrix([[1,0,0],[0,0,0],[0,0,-1]])
    XTab = MakeSiteOperator(X,n)
    YTab = MakeSiteOperator(Y,n)
    ZTab = MakeSiteOperator(Z,n)

    Sdot = np.dot(XTab[0],XTab[1])+np.dot(YTab[0],YTab[1])+np.dot(ZTab[0],ZTab[1])
    H = 1/3 * (np.identity(Sdot.shape[0])) + 1/2* Sdot + 1/6 * (Sdot**2)
    for i in range(1,len(XTab)-1):
        Sdot =  np.dot(XTab[i],XTab[i+1])+np.dot(YTab[i],YTab[i+1])+np.dot(ZTab[i],ZTab[i+1])
        H += 1/3 * (np.identity(Sdot.shape[0])) + 1/2* Sdot + 1/6 * (Sdot**2)
    return H
