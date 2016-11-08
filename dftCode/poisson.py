"""This module contains 1 line poisson funtion used to solve the poisson equation."""
import numpy as np

from dftCode.fourierBasis import Ope, Lope, LinvOpe, cJ, cI

def phi(inp,s,R):
    """Poisson solver in a single line of code."""
    return cI(LinvOpe(-4.0*np.pi*Ope(cJ(inp,s,R),R),s,R),s,R)



