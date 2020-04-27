import numpy as np

def possible_AGSP(H,n):
    Id = np.identity(H.shape[0])
    eval = np.linalg.eigvalsh(H)
    eval_max = eval[-1]
    eval_gr = eval[0]
    if abs(eval_gr)>eval_max:
        max = abs(eval_gr)
    else:
        max = eval_max
    if np.isclose(eval_gr,0):
        print("frustration free!")
        K = Id - H/max
    elif eval_gr > 0:
        K = (1/(1-eval_gr/max))*(Id - H/max)
    elif (eval_gr < 0 ) and (eval_max != max):
        K = 1/2*(Id - H/max)
    elif (eval_gr < 0 ) and (eval_max == max):
        K = (1/(1-eval_gr/max))*(Id - H/max)
    else:
        print("Something wrong happens. Wierd case :<")
    retK = np.linalg.matrix_power(K,n)
    return retK

def nullity(M,k):
    eval = np.linalg.eigvalsh(M)
    dim = M.shape[0]
    eval = eval[k]
    N = M-eval*np.identity(dim)
    gmulti = dim-np.linalg.matrix_rank(N)
    return gmulti

def project_random_vector(H,n):
    eigen = np.linalg.eigh(H)
    evec = eigen[1]
    gr = evec[:,0]
    nlty = nullity(H,0)

    rrstate = np.kron(np.random.rand(2**(n//2),1),np.random.rand(2**(n-n//2),1))
    rstate = np.kron(np.random.rand(2**(n//2),1),np.random.rand(2**(n-n//2),1))
    i = 0
    if nlty == 1:
        while not (np.all(np.isclose(gr,rstate)) or np.all(np.isclose(gr,-rstate))):
            i+=10
            K = possible_AGSP(H,i)
            rstate = K.dot(rstate)
            rstate = rstate/np.linalg.norm(rstate)
    elif (nlty == 2) or (nlty == 4):
        while not (np.all(np.isclose(rstate,rrstate))):
            i+=10
            K = possible_AGSP(H,i)
            rrstate = rstate
            rstate = K.dot(rstate)
            rstate = rstate/np.linalg.norm(rstate)
    else:
        print("Unexpected error: nullity of the system is " + str(nlty))
        raise
    print("Number of Qubits: "+str(n)+"    Power of AGSP: "+str(i))
    return i
