from Schmidt_Decomposition import *

from XXZ_Classical_Hamiltonian import *

def refillvec(v,n):
    gr_basis = generateBasisRep(n,"Gr")
    all_basis = generateBasisRep(n)
    index = [all_basis.index(gr_basis[i]) for i in range(len(gr_basis))]
    ret_array = np.zeros(2**n)
    for i in range(len(index)):
        ret_array[index[i]] = v[i]
    return ret_array

def VNEntropy(H,n,dA,dB):
    eval,evec = np.linalg.eigh(H)
    ground_evec = evec[:,0]

    if H.shape[0] != 2**n:
        ground_evec = refillvec(ground_evec,n)
    else:
        ground_evec = np.transpose(ground_evec)
    sd = SD_discrete(ground_evec,"vector",dA,dB)

    # vnEntro = sum([-sd[i][0]**2*2*np.log2(sd[i][0]) for i in range(len(sd)) if not np.isclose(sd[i][0],0)])
    vnEntro = sum([-np.around(sd[i][0], decimals = 10)**2*2*np.log2(np.around(sd[i][0], decimals = 10)) for i in range(len(sd)) if not np.isclose(sd[i][0],0)])

    # vnEntro = sum([-sd[i][0]**2*2*np.log2(sd[i][0]) for i in range(len(sd))])
    return vnEntro

def SDCoeff(H,n,dA,dB):
    eval,evec = np.linalg.eigh(H)
    ground_evec = evec[:,0]

    if H.shape[0] != 2**n:
        ground_evec = refillvec(ground_evec,n)
    sd = SD_discrete(ground_evec,"vector",dA,dB)

    sd_coeff = [sd[i][0] for i in range(len(sd))]
    return sd_coeff
