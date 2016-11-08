import numpy as np
import pytest
from dftCode.geometry import _M, _N, gen_r_G_G2
from dftCode.charge_dis import gaussian,charge_dis


def test_gaussian():
    """ Tests the gaussian charge distribution routine in charge_dis.py
    """
    assert np.allclose(gaussian([0.75,0.5],[1,-1],np.linspace(0,1,10)),np.array([-0.35744565, -0.34669986, -0.31613924, -0.27038447, -0.21590651,
       -0.15959883, -0.10743921, -0.06358218, -0.03003532, -0.00686962])) == True
    assert np.allclose(gaussian([0.75,0.5,0.3],[1,-1,0.2],np.linspace(0,1,10)),np.array([ 0.11287757,  0.09244664,  0.04133802, -0.01668859, -0.05894094,
       -0.07493042, -0.06762259, -0.04725793, -0.02420049, -0.00505139]))

def test_charge_dis():
    """ Tests the charge_dis subroutine in charge_dis.py                                                                                                 
    """
    assert np.allclose(charge_dis([2,2,2],np.diag([2,2,2]),[0.75,0.5],[1,-1]),np.array([ 0.00919842,  0.01613367,  0.01613367, -0.00686962,  0.01613367,
      -0.00686962, -0.00686962, -0.35744565])) == True




