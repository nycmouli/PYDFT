"""Plane wave basis for the DFT solvers. Exposes the generic linear
algebra functions required by the other solvers and methods.
"""


# For performing linear algebra operations using matrices
from numpy import linalg as LA
from numpy.linalg import inv
import operator
# for Generating tuples list.
import itertools 


from dftCode import msg
from dftCode.geometry import gen_r_G_G2

import numpy as np


def Ope(inp,R):
    """Outputs the overlap matrix. In our case here we are specific to plane wave basis.
    Args:
        input:
            inp (numpy array): The overlap matrix operator acts on this input.
            R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
    """
    return np.linalg.det(R)*np.dot(np.identity(len(inp)),inp)   


def Lope(inp,s,R):
    """Lope is the L operator as defined in Tomas arias Lecture.
     Note:Works only for plane wave basis.
    Args:
        input:
            inp (numpy array): The overlap matrix operator acts on this input.
            s=[a,b,c] => Defines the number of sample points along the
                        lattice vectors.  (a,b,c) are any integers.
            R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
    """
    if len(gen_r_G_G2(s,R,"G2")) == len(inp):
        return -np.linalg.det(R)*np.dot(np.diag(gen_r_G_G2(s,R,"G2")),inp)
    else:
        return "Error: Input vector size is not same as G2"

def LinvOpe(inp,s,R):
    """LinvOpe is the inverse of L operator as defined in Tomas arias Lecture.
     Note:Works only for plane wave basis.
    Args:
        input:
            inp (numpy array): The overlap matrix operator acts on this input.
            s=[a,b,c] => Defines the number of sample points along the
                        lattice vectors.  (a,b,c) are any integers.
            R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
    """
    if len(gen_r_G_G2(s,R,"G2")) == len(inp):
        Lele=-np.linalg.det(R)*np.diag(gen_r_G_G2(s,R,"G2"))
        Lele[0][0]=1
        Leleinv=inv(Lele)
        Leleinv[0][0]=0
        return np.dot(Leleinv,inp)
    else:
        return "Error: Input vector size is not same as G2"

def cI(inp,s,R,p=None):
    """cI is the fourier series expansion coeff Matrix. It is the forward transform matrix for the problem.
        Note:Works only for plane wave basis.
        Args:
        input:
            inp (numpy array): The overlap matrix operator acts on this input.
            s=[a,b,c] => Defines the number of sample points along the
                        lattice vectors.  (a,b,c) are any integers.
            R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
            p (default): Fast fourier transform. #Written with Wiley's code help.
                p='s' => slow fourier transform.
    """
    if p is None:
        f=np.fft.fftn(inp.reshape(s,order="F")).reshape(np.prod(s),order="F")
    elif p is "s": 
        f=np.dot(np.exp(1j*np.dot(gen_r_G_G2(s,R,"G"),np.transpose(gen_r_G_G2(s,R,"r")))),inp)
    return f

def cJ(inp,s,R,p=None):
    """cI is the fourier series expansion coeff Matrix. You can get the fast fourier transform too.
      Note:Works only for plane wave basis.
    Args:
        input:
            inp (numpy array): The overlap matrix operator acts on this input.
            s=[a,b,c] => Defines the number of sample points along the
                        lattice vectors.  (a,b,c) are any integers.
            R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
    """
    if p is None:
        f=np.fft.ifftn(inp.reshape(s,order="F")).reshape(np.prod(s),order="F")
        return f
    elif p is "s":
        f=np.exp(1j*np.dot(gen_r_G_G2(s,R,"G"),np.transpose(gen_r_G_G2(s,R,"r"))))
        return (1./np.prod(s))*np.dot(np.transpose(f.conjugate()),inp)


