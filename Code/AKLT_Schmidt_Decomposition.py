from Schmidt_Decomposition import *

def VNEntropy(H,n,dA,dB):
    eval,evec = np.linalg.eigh(H)
    ground_evec = evec[:,0]

    sd = SD_discrete3(ground_evec,"vector",dA,dB)

    vnEntro = sum([-np.around(sd[i][0], decimals = 10)**2*2*np.log2(np.around(sd[i][0], decimals = 10)) for i in range(len(sd)) if not np.isclose(sd[i][0],0)])

    return vnEntro
