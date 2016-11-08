import numpy as np

from dftCode.geometry import gen_r_G_G2


def gaussian(sig,coeff,x):
    """ Defining Gaussian charge distribution at point x in 3D space.
    Args:
        inputs: 
            sig => sigmas for each Gaussian.
            coeff => coefficients for each gaussian funtion.
                Note: All the gaussians will be summed up.
            x (np.array()): List of input points 
                Note: x in 3D space is x=Sqrt(a^2+b^2+c^2) where
                {a,b,c} are the cartesian coordinates.
            
        Output: Charge distribution at points x. 
        
    """
    g=lambda x,sigma: (np.exp(-(x**2)/(2*sigma**2)))/(2*np.pi*sigma**2)**(3./2)
    n=lambda x: np.sum([coeff[i]*g(x,sig[i]) for i in range(len(sig))],axis=0)
    if len(sig)!=len(coeff):
        return "Error: Length of 'sig' not equal to length of 'coeff'."
    else:
        return n(x)


def charge_dis(s,R,sig,coeff):
    """Generates the charge distribution at each sample point for the unit cell.
        We change the midpoint of the unitcell to origin here.
    
    Args:
        input:
            R=[a,b,c] => Defines the lattice vectors. (a,b,c) are any +ve integers.
            s=[a,b,c] => Defines the number of sample points along the
                        lattice vectors.  (a,b,c) are any integers.
    """
    # There is a problem with sum()
    mid = [sum(R[i])/2. for i in range(len(R))] #sum(R,axis=1)/2.
    #Note: Need to check if this mid is right for non diagnol R vectors.
    #dr=np.sqrt(sum((gen_r_G_G2(s,R,"r")-mid)**2,axis=1))
    ss=(gen_r_G_G2(s,R,"r")-mid)**2
    dr=np.sqrt([sum(ss[i]) for i in range(len(ss))])
    return gaussian(sig,coeff,dr)
        
