#!/usr/bin/python

from dftCode import msg
import numpy as np
import csv


def examples():
    """Prints examples of using the script to the console using colored output.                                                   
    """
    script = "Code to compute DFT energies of a crystal structure."
    explain = ("This code produces a numerical solution for the energy of a system of "
               "atoms. As of now we have only the poisson solver.")
    contents = [(("Find the charge density for a simple cubic cell with lattice parameter of 6 "
                  "with 6 divisions along each lattice vector."),
                 "dftSolve.py 6 -s [6,6,6]",
                 "This prints the value of potential evaluated at all the sample points "
                 "to the screen")]
    required = ("REQUIRED: The lattice parameter to be specified `a`.")
    output = ("RETURNS: A potential file named `potential.csv` if -poisson is specified.")
    details = (".")
    outputfmt = ("")

    msg.example(script, explain, contents, required, output, outputfmt, details)


# script options is for arg parse. I'm not intentding to use it. But I dont want to delete it now.

script_options = {
    "-la": dict(default=1., type=float,
              help=("Lattice parameter for the crystal structure.")),
    "-crystal": dict(default="sc", type=str,
                     help=("The type of primitive cell to use options are (sc)")),
    "-s": dict(default=[6,6,6], nargs="+",
              help=("Number of sampling points along each basis vector.")),
    "-poisson": dict(action="store_true",
                     help=("Prints the solution to the poisson equation to the screen.")),
    
     }

# _parser_options is also for arg parse. I'm not intending to use it. I will use config parse.

def _parser_options():
    """Parses the options and arguments from the command line."""
    #We have two options: get some of the details from the config file,                                                  
    import argparse
    from dftCode import base

    pdescr = "Numerical DFT code."
    parser = argparse.ArgumentParser(parents=[base.bparser], description=pdescr)
    for arg, options in script_options.items():
        parser.add_argument(arg, **options)

    

    args = base.exhandler(examples, parser)

    if args is None:
        return

    return args # pragma: no cover 


# The run below runs with args. I wrote a better config parse code below. So not intented to use this function.
def run(args):
    """Runs the code to find the energy of the crystal using DFT.
    """

    if args is None:
        print "No args given"
        exit

    if args["poisson"]:

        from dftCode.fourierBasis import Ope, Lope, LinvOpe, cJ, cI
        from dftCode.poisson import phi
        from dftCode.charge_dis import charge_dis
        
        R = args["la"]*np.array([[1,0,0],[0,1,0],[0,0,1]])
        s = [int(i) for i in args["s"]]
        
        sig=[0.75,0.5]
        coeff=[-1,1]

        # Analytic energy for Poisson
        Uanalytic=((1/sig[0] + 1/sig[1])/2 -np.sqrt(2)/np.sqrt(sig[0]**2+sig[1]**2))/np.sqrt(np.pi)

        # Numerical energy solved using 1 line poisson solver.

        Unumerical=0.5*np.real(np.dot(np.conjugate(cJ(np.real(phi(charge_dis(s,R,sig,coeff),s,R)),s,R)),Ope(cJ(charge_dis(s,R,sig,coeff),s,R),R)))

        
        print "Uanalytic =", Uanalytic
        print"Unumerical =" , Unumerical

        return 0



def ConfigSectionMap(section):

    # The code has to work for anyone who downloads.                                                                                                                      
    # So, DATA_PATH gives the path to config file.                                                                                                                       
    import os
    this_dir, this_filename = os.path.split(__file__)
    tmp=this_dir.split("/")
    actual_path="/".join(tmp[:-1])
    DATA_PATH = os.path.join(actual_path, "InpDFT", "input_dft.cfg")



    # Writing config parser part with help from https://wiki.python.org/moin/ConfigParserExamples                                                                         
    import ConfigParser
    Config = ConfigParser.ConfigParser()
    Config.read(DATA_PATH)

    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


def checkFloat(i):
    try:
        float(i)
        return True
    except:
        print("Warning: Check your input file. 'Spaces' or any other 'wrong input' are not yet handled.")
        return False


def run_config():
    """Runs the code using "input_dft.cfg" to find the energy of the crystal using DFT    
    """


    from dftCode.fourierBasis import Ope, Lope, LinvOpe, cJ, cI
    from dftCode.poisson import phi
    from dftCode.charge_dis import charge_dis


    geo = ConfigSectionMap("geometry")['geo']
    rIn = ConfigSectionMap("geometry")['r']
    sIn = ConfigSectionMap("geometry")['s']
    
    if(geo!="simpleCubic"):
        print("Only simple Cubic geometry is programmed now. \n Please pass in geo=simpleCubic in config file.")
    else:

        tmp=rIn.split(" ")
        R=[int(i) for i in tmp if checkFloat(i)] 
        R = R*np.array([[1,0,0],[0,1,0],[0,0,1]])
        #print(R)

        tmp=sIn.split(" ")
        s=[int(i) for i in tmp if checkFloat(i)]
        #print(s)

    
    char_dis = ConfigSectionMap("chargeDis")['char_dis']
    sigIn = ConfigSectionMap("chargeDis")['sig']
    coeffIn=ConfigSectionMap("chargeDis")['coeff']


    if(char_dis!="gaussian"):
        print("Only Gaussian distribution is programmed now. \n Please pass in 'char_dis=gaussian' in config file.")
    else:

        tmp=sigIn.split(" ")
        sig=[float(i) for i in tmp if checkFloat(i)]
        #print(sig)

        tmp=coeffIn.split(" ")
        coeff=[int(i) for i in tmp if checkFloat(i)]
        #print(coeff)

        # Analytic energy for Poisson                                                                                                                                      
        Uanalytic=((1/sig[0] + 1/sig[1])/2 -np.sqrt(2)/np.sqrt(sig[0]**2+sig[1]**2))/np.sqrt(np.pi)

        # Numerical energy solved using 1 line poisson solver.                                                                                                           
        Unumerical=0.5*np.real(np.dot(np.conjugate(cJ(np.real(phi(charge_dis(s,R,sig,coeff),s,R)),s,R)),Ope(cJ(charge_dis(s,R,sig,coeff),s,R),R)))

        print "Uanalytic =", Uanalytic
        print"Unumerical =" , Unumerical
        
    return 0

     
if __name__ == '__main__': # pragma: no cover
    #run(_parser_options()) # As of now I do not want to run with arg parse.
    run_config() # Runs with config file.
