import numpy as np
import pytest
from dftCode.fourierBasis import Ope, Lope, LinvOpe, cI, cJ
from dftCode.geometry import _M, _N, gen_r_G_G2

def test_Ope():
    """Tests the Ope() subroutine in fourierBasis.py
    """

    assert np.allclose(Ope([2,3,4,5],np.diag([2,2,2])),np.array([ 16.,  24.,  32.,  40.])) == True
    assert np.allclose(Ope([1,2,3,4,5,5.5,6,3],np.diag([2,1,6])),np.array([ 12.,  24.,  36.,  48.,  60.,  66.,  72.,  36.])) == True



def test_Lope():
    """Tests the Lope() subroutine in fourierBasis.py
      
    Function: Lope(inp,s,R)
    
    input:
            inp (numpy array): The overlap matrix operator acts on this input.
            s=[a,b,c] => Defines the number of sample points along the
                        lattice vectors.  (a,b,c) are any integers.
            R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
    """

    assert Lope([2,3,4,5],[2,2,2],np.diag([2,2,2])) == 'Error: Input vector size is not same as G2'
    assert np.allclose(Lope(gen_r_G_G2([2,2,2],np.diag([2,2,2]),'G2'),[2,2,2],np.diag([2,2,2])),np.array([   -0.        ,  -779.27272827,  -779.27272827, -3117.09091309,
       -779.27272827, -3117.09091309, -3117.09091309, -7013.45455445])) == True
    
def test_LinvOpe():
    """ Testing the LinvOpe operator in fourierBasis.py

    Funtion: LinvOpe
    
    LinvOpe is the inverse of L operator as defined in Tomas arias Lecture.
     Note:Works only for plane wave basis.
    Args:
        input:
            inp (numpy array): The overlap matrix operator acts on this input.
            s=[a,b,c] => Defines the number of sample points along the
                        lattice vectors.  (a,b,c) are any integers.
            R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
    """
    assert LinvOpe([2,3,4,5],[2,2,2],np.diag([2,2,2])) == 'Error: Input vector size is not same as G2'
    assert np.allclose(LinvOpe(gen_r_G_G2([2,2,2],np.diag([2,2,2]),'G2'),[2,2,2],np.diag([2,2,2])),np.array([ 0.   , -0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125]))


def test_cI():
    """ Testing the cI subroutine in fourierBasis.py
    
    Function: cI
    cI is the fourier series expansion coeff Matrix. It is the forward transform matrix for the problem.
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

    assert np.allclose(cI(gen_r_G_G2([2,2,2],np.diag([3,3,3]),'r').T[1],[2,2,2],np.diag([3,3,3])),np.array([ 6.+0.j,  0.+0.j, -6.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,0.+0.j]))
    assert np.allclose(cI(gen_r_G_G2([2,2,2],np.diag([3,3,3]),'r').T[1],[2,2,2],np.diag([3,3,3]),'s'),np.array([ 6. +0.00000000e+00j,  0. +3.67394040e-16j, -6. +7.34788079e-16j,
                                                                    0. -3.67394040e-16j,  0. +3.67394040e-16j,  0. -2.46519033e-32j,0. -3.67394040e-16j,  0. -2.46519033e-32j]))
def test_cJ():
    """
    This funtion tests the cJ funtion in fourierBasis.py

    Funtion: cJ
    cJ is the fourier series expansion coeff Matrix. You can get the fast fourier transform too.
      Note:Works only for plane wave basis.
    Args:
        input:
            inp (numpy array): The overlap matrix operator acts on this input.
            s=[a,b,c] => Defines the number of sample points along the
                        lattice vectors.  (a,b,c) are any integers.
            R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
    """
    
    assert np.allclose(cJ(gen_r_G_G2([2,2,2],np.diag([3,3,3]),'r').T[1],[2,2,2],np.diag([3,3,3])),np.array([ 0.75+0.j,  0.00+0.j, -0.75+0.j,  0.00+0.j,  0.00+0.j,  0.00+0.j,0.00+0.j,  0.00+0.j]))
    assert np.allclose(cJ(gen_r_G_G2([2,2,2],np.diag([3,3,3]),'r').T[1],[2,2,2],np.diag([3,3,3]),'s'),np.array([ 0.75 +0.00000000e+00j,  0.00 -4.59242550e-17j,
                                                                                                                  -0.75 -9.18485099e-17j,  0.00 +4.59242550e-17j,
                                                                                                                  0.00 -4.59242550e-17j,  0.00 +3.08148791e-33j,
                                                                                                                  0.00 +4.59242550e-17j,  0.00 +6.16297582e-33j]))
