import numpy as np
import pytest
from dftCode.geometry import _M, _N, gen_r_G_G2
import sys


def test_M():
    """Tests that the _M subroutine works.                                                                                                                   
    """
    assert _M([2,2,2])==[[0, 0, 0],[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    assert _M([3,2,1])==[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]]


def test_N():
    """Tests that the  _N subroutine works.
    This is another way to test arrays. using np.allclose
    """
    assert np.allclose(_N([3,2,1]),np.array([[ 0,  0,  0],[ 0,  1,  0],[ 1,  0,  0],[ 1,  1,  0],[-1,  0,  0],[-1,  1,  0]])) == True
    assert np.allclose(_N([2,2,2]),np.array([[0, 0, 0],
                                             [0, 0, 1],
                                             [0, 1, 0],
                                             [0, 1, 1],
                                             [1, 0, 0],
                                             [1, 0, 1],
                                             [1, 1, 0],
                                             [1, 1, 1]]))

def test_gen_r_G_G2():
    """ Tests weather gen_r_G_G2 subroutine works
    """
    assert np.allclose(gen_r_G_G2([2,2,2],np.diag([2,2,2]),'r'),np.array([[ 0.,  0.,  0.],
                                                                           [ 0.,  0.,  1.],
                                                                           [ 0.,  1.,  0.],
                                                                           [ 0.,  1.,  1.],
                                                                           [ 1.,  0.,  0.],
                                                                           [ 1.,  0.,  1.],
                                                                           [ 1.,  1.,  0.],
                                                                           [ 1.,  1.,  1.]]))
    assert np.allclose(gen_r_G_G2([2,2,2],np.diag([2,2,2]),'G'),np.array([[ 0.        ,  0.        ,  0.        ],
                                                                           [ 0.        ,  0.        ,  3.14159265],
                                                                           [ 0.        ,  3.14159265,  0.        ],
                                                                           [ 0.        ,  3.14159265,  3.14159265],
                                                                           [ 3.14159265,  0.        ,  0.        ],
                                                                           [ 3.14159265,  0.        ,  3.14159265],
                                                                           [ 3.14159265,  3.14159265,  0.        ],
                                                                           [ 3.14159265,  3.14159265,  3.14159265]])) 
    assert np.allclose(gen_r_G_G2([2,2,2],np.diag([2,2,2]),'G2'),np.array([  0.       ,   9.8696044,   9.8696044,  19.7392088,   9.8696044,
                                                                              19.7392088,  19.7392088,  29.6088132]))
