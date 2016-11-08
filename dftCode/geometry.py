
import numpy as np
# For performing linear algebra operations using matrices
from numpy import linalg as LA
from numpy.linalg import inv
import operator
# for Generating tuples list.
import itertools 


def _M(s):
    """Defining the 'M' matrix.
    'M' has coordinates in matrix form for each sample point in real space.
    We use M to get list of real space sample  points 'r'
    r=M(Diag(S))^(-1)R^(Transpose)
    
     Args:
        input:
            s => number of sample points along the lattice vectors.
        output: 
            M matrix. 
    """
    M = tuple(itertools.product(*[range(0,s[0]),range(0,s[1]),range(0,s[2])]))
    M=[list(i) for i in M]
    return M



def _N(s):
    """Defining the 'N' matrix.
    'N' has coordinates in matrix form for each sample point in reciprocal
    space.
    We use N to get list of reciprocal lattice vectos 'G'
    G= 2*pi*N*R^(inverse)
    
    Args:
        input:
            s => number of sample points along the lattice vectors.
        output: 
            N matrix. 
    """
    M=_M(s)
    m1=np.transpose(M)[0]
    m2=np.transpose(M)[1]
    m3=np.transpose(M)[2]
    n1=[ m1[i]-s[0] if m1[i] > s[0]/2 else m1[i] for i in range(len(m1)) ]
    n2=[ m2[i]-s[1] if m2[i] > s[1]/2 else m2[i] for i in range(len(m2)) ]
    n3=[ m3[i]-s[2] if m3[i] > s[2]/2 else m3[i] for i in range(len(m3)) ]
    N=np.transpose([n1,n2,n3])
    return N

def gen_r_G_G2(s,R,p=None):
    """Generates r, G and G2 matrices.
    
    Args: input:
           s => number of sample points along the lattice vectors. 
           R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
           p = 'r', 'G' or 'G2'(square of G matrix) 
           
          Output:
           p='r' => Outputs r matrix.
           p='G' => Outputs G matrix.
           p='2' => Outputs G2 matrix.
       
       Example: s = [15,15,15]
                R= np.diag([6,6,6])
                _gen_r_G_G2(s,R,"r")
    """
    if p is None:
        return "Please pass a third argument asking for 'r', 'G' or 'G2'."
    if p is "r":
        r=np.dot(np.dot(_M(s),inv(np.diag(s))),np.transpose(R))
        return r
    if p is "G":
        G=2*np.pi*np.dot(_N(s),inv(R))
        return G
    if p is "G2":
        G=2*np.pi*np.dot(_N(s),inv(R))
        G2=np.sum(np.transpose(G*G),axis=0) 
        return G2
    else:
        return "Error: Passed an argument other than r, G or G2."
