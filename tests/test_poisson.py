import numpy as np
import pytest
from dftCode.fourierBasis import Ope, Lope, LinvOpe, cI, cJ
from dftCode.geometry import _M, _N, gen_r_G_G2
from dftCode.charge_dis import charge_dis
from dftCode.poisson import phi



def test_phi():
    """
     Tests the oneline poisson solver funtion.
    
    Poisson solver in a single line of code."""
    
    assert np.allclose(phi(charge_dis([2,2,2],np.diag([2,2,2]),[0.75,0.5],[1,-1]),[2,2,2],np.diag([2,2,2])),np.array([ 0.11648337+0.j,  0.07467416+0.j,  0.07467416+0.j, -0.01777448+0.j,
    0.07467416+0.j, -0.01777448+0.j, -0.01777448+0.j, -0.28718243+0.j]))
