import numpy as np

def generateBasisRep(n,desc="All"):
    dim = 2**n
    total_basis = [format(i,"0"+str(n)+"b") for i in range(dim)]
    if desc == "Gr":
        if n % 2 == 0:
            ret_basis = [vec for vec in total_basis if vec.count('0') == n//2]
        else:
            ret_basis = [vec for vec in total_basis if ((vec.count('0') == n//2) or  (vec.count('0') == n//2+1))]
    else:
        ret_basis = total_basis
    return ret_basis

def swap(s,i,j):
    ls = list(s)
    ls[i],ls[j]=ls[j],ls[i]
    return ''.join(ls)

def generateHamiltonianXXZGr(n,delta,J):
    basis = generateBasisRep(n,"Gr")
    dim = len(basis)
    ret_matrix = [0]*dim
    for i in range(dim//2):
        vec = basis[i]
        count_double = 0
        ret_vec = [0]*dim
        for j in range(n-1):
            if vec[j] == vec[j+1]:
                count_double += 1
            else:
                new_vec = swap(vec,j,j+1)
                index = basis.index(new_vec)
                ret_vec[index] = 2
        ret_vec[i] = delta * (count_double - ((n-1) - count_double))
        ret_matrix[i] = ret_vec
        ret_matrix[(dim-1)-i] = ret_vec[::-1]
    ret_array = np.array(ret_matrix)*J/4
    return ret_array

# Classical Method to generate a Hamiltonian matrix
def KroneckerMultiProduct(s):
    retval = s[0]
    for i in range(1,len(s)):
        retval = np.kron(retval,s[i])
    return retval

def MultiId(M):
    Id = np.matrix([[1,0],[0,1]])
    return KroneckerMultiProduct([Id]*M)

def MakeSiteOperator(U,Num):
    ret_ls = [np.kron(U,MultiId(Num-1))]
    for i in range(2,Num):
        ret_ls.append(KroneckerMultiProduct([MultiId(i-1),U,MultiId(Num-i)]))
    ret_ls.append(np.kron(MultiId(Num-1),U))
    return ret_ls

def generateHamiltonianXXZ(n,delta,J):
    X = np.matrix([[0,1],[1,0]])
    Y = np.matrix([[0,-1j],[1j,0]])
    Z = np.matrix([[1,0],[0,-1]])
    XTab = MakeSiteOperator(X,n)
    YTab = MakeSiteOperator(Y,n)
    ZTab = MakeSiteOperator(Z,n)
    H = J/4 * (np.dot(XTab[0],XTab[1])+np.dot(YTab[0],YTab[1])+delta*np.dot(ZTab[0],ZTab[1]))
    for i in range(1,len(XTab)-1):
        H += J/4 * (np.dot(XTab[i],XTab[i+1])+np.dot(YTab[i],YTab[i+1])+delta*np.dot(ZTab[i],ZTab[i+1]))
    return H
