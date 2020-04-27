import operator as op
from functools import reduce
import numpy as np

def generateBasisVector(n):
    ones = np.ones(n)
    return np.diag(ones)

def SD_discrete(state,mode,dA,dB):
    if mode == "vector":
        a = np.reshape(state,(2**dA,2**dB))
    else:
        a = state

    basisA = generateBasisVector(2**dA)
    basisB = generateBasisVector(2**dB)
    (u,d,vt) = np.linalg.svd(a)
    v = np.transpose(vt)

    ilambda = d
    iA = [sum([u[j,i]*basisA[j] for j in range(2**dA)]) for i in range(2**dA)]
    iB = [sum([v[i,k]*basisB[k] for k in range(2**dB)]) for i in range(2**dB)]

    ret = [(ilambda[i],iA[i],iB[i]) for i in range(len(ilambda))]
    return ret
