#!python

import numpy as np
import matplotlib.pyplot as plt
import sys, os

import cellconstructor as CC, cellconstructor.Units

import time
import cellconstructor.Methods

import sscha, sscha.SchaMinimizer
import sscha.Ensemble
import sscha.Relax
import sscha.Utilities
import argparse


"""
This is part of the program python-sscha
Copyright (C) 2018  Lorenzo Monacelli

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>. 
"""

DESCRIPTION = '''
This program plots the results of a SSCHA minimization 
saved with the Utilities.IOInfo class.

It plots both frequencies and the free energy minimization.

Many files can be specified, in this case, they are plotted one after the other,
separated by vertical dashed lines.

Usage:
 
 >>> sscha-plot-data.x file1  ...

The code looks for data file called file.dat and file.freqs, 
containing, respectively, the minimization data and the auxiliary frequencies.

These files are generated only by python-sscha >= 1.2.

'''


def plot_data():
    print(DESCRIPTION)

    if len(sys.argv) < 2:
        ERROR_MSG = '''
Error, you need to specify at least one file.
Exit failure!
'''
        print(ERROR_MSG)
        return 1
        #raise ValueError(ERROR_MSG)
    
    plt.rcParams["font.family"] = "Liberation Serif"
    LBL_FS = 12
    DPI = 120
    TITLE_FS = 15

    freqs_files = [sys.argv[x] + '.freqs' for x in range(1, len(sys.argv))]
    minim_files = [sys.argv[x] + '.dat' for x in range(1, len(sys.argv))]

    # Check if all the files exist
    plot_frequencies = None
    plot_minim = None
    for f, m in zip(freqs_files, minim_files):
        if os.path.exists(f):
            if plot_frequencies is None:
                plot_frequencies = True
            if not plot_frequencies:
                raise IOError("Error, file {} found, but not others .freqs".format(f))
        else:
            if plot_frequencies:
                raise IOError("Error, file {} not found.".format(f))
            plot_frequencies = False


        if os.path.exists(m):
            if plot_minim is None:
                plot_minim = True
            if not plot_minim:
                raise IOError("Error, file {} found, but not others .freqs".format(m))
        else:
            if plot_minim:
                raise IOError("Error, file {} not found.".format(m))
            plot_minim = False

    if not plot_frequencies and not plot_minim:
        print("Nothing to plot, check if the .dat and .freqs files exist.")
        exit()

    if plot_minim:
        print("Preparing the minimization data...") 
        minim_data = np.concatenate([np.loadtxt(f) for f in minim_files])

        # Insert the x axis in the plotting data
        xsteps = np.arange(minim_data.shape[0])
        new_data = np.zeros(( len(xsteps), 8), dtype = np.double)
        new_data[:,0] = xsteps
        new_data[:, 1:] = minim_data
        minim_data = new_data
        
        fig_data, axarr = plt.subplots(nrows=2, ncols = 2, sharex = True, dpi = DPI)
        
        # Plot the data
        axarr[0,0].fill_between(minim_data[:,0], minim_data[:,1] - minim_data[:, 2]*.5 ,
                                minim_data[:, 1] + minim_data[:, 2] * .5, color = "aquamarine")
        axarr[0,0].plot(minim_data[:,0], minim_data[:,1], color = "k")
        axarr[0,0].set_ylabel("Free energy / unit cell [meV]", fontsize = LBL_FS)


        axarr[0,1].fill_between(minim_data[:,0], minim_data[:,3] - minim_data[:, 4]*.5 ,
                                minim_data[:, 3] + minim_data[:, 4] * .5, color = "aquamarine")
        axarr[0,1].plot(minim_data[:,0], minim_data[:,3], color = "k")
        axarr[0,1].set_ylabel("FC gradient", fontsize = LBL_FS)

        axarr[1,1].fill_between(minim_data[:,0], minim_data[:,5] - minim_data[:, 6]*.5 ,
                                minim_data[:, 5] + minim_data[:, 6] * .5, color = "aquamarine")
        axarr[1,1].plot(minim_data[:,0], minim_data[:,5], color = "k")
        axarr[1,1].set_ylabel("Structure gradient", fontsize = LBL_FS)
        axarr[1,1].set_xlabel("Good minimization steps", fontsize = LBL_FS)


        axarr[1,0].plot(minim_data[:,0], minim_data[:,7], color = "k")
        axarr[1,0].set_ylabel("Effective sample size", fontsize = LBL_FS)
        axarr[1,0].set_xlabel("Good minimization steps", fontsize = LBL_FS)
        fig_data.tight_layout()
    



    if plot_frequencies:
        print("Plotting the frequencies")

        # Load all the data
        freqs_data = np.concatenate([np.loadtxt(f) for f in freqs_files])
        # Now plot the frequencies
        fig_freqs = plt.figure(dpi = DPI)
        ax = plt.gca()

        N_points, Nw = np.shape(freqs_data)

        for i in range(Nw):
            ax.plot(freqs_data[:, i] * CC.Units.RY_TO_CM)

        ax.set_xlabel("Good minimization steps", fontsize = LBL_FS)
        ax.set_ylabel("Frequency [cm-1]", fontsize = LBL_FS)
        ax.set_title("Frequcency evolution", fontsize = TITLE_FS)
        fig_freqs.tight_layout()

    plt.show()
    return 0
 
def main():
    PROG_NAME = "sscha"
    VERSION = "1.5"
    DESCRIPTION = r"""



    * * * * * * * * * * * * * * * * * * * * * * * * * * *
    *                                                   *
    * STOCHASTIC SELF-CONSISTENT HARMONIC APPROXIMATION *
    *                                                   *
    * * * * * * * * * * * * * * * * * * * * * * * * * * *



    Wellcome to the SSCHA code. 
    This stand-alone script performs SSCHA minimization 
    according to the options parsed from the input file.
    Note, you can also run the SSCHA with the python-APIs. 
    The python-APIs give access to much more customizable options,
    not available with the stand-alone app.

    Refer to the documentation for more details.


    """


    # Define a custom function to raise an error if the file does not exist
    def is_valid_file(parser, arg):
        if not os.path.exists(arg):
            parser.error("The file %s does not exists!" % arg)
        else:   
            return arg

    # Prepare the parser for the command line
    parser = argparse.ArgumentParser(prog = PROG_NAME,
                                     description = DESCRIPTION,
                                     formatter_class = argparse.RawTextHelpFormatter)

    # Add the arguments
    parser.add_argument("-i", help="Path to the input file",
                        dest="filename", required = True, metavar = "FILE",
                        type = lambda x : is_valid_file(parser, x))
    parser.add_argument("--plot-results", 
                        help="Plot the results",
                        dest="plot", action = "store_true")

    parser.add_argument("--save-data", dest="save_data", metavar = "FILE",
                        help = "Save the minimization details")


    # Get the arguments from the command line
    args = parser.parse_args()

    # Get the input filename and initialize the sscha minimizer.
    minim = sscha.SchaMinimizer.SSCHA_Minimizer()

    # Open the input file
    fp = open(args.filename, "r")
    line_list = fp.readlines()
    fp.close()

    # Check if the mandatory arguments are defined
    namelist = CC.Methods.read_namelist(line_list)
    if not sscha.SchaMinimizer.__SCHA_NAMELIST__ in namelist.keys():
        print("Error, %s namelist required." % sscha.SchaMinimizer.__SCHA_NAMELIST__)
        return 1

    # Check if the relax is hypothized
    if sscha.Relax.__RELAX_NAMESPACE__ in namelist.keys():
        relax = sscha.Relax.SSCHA()
        print ("Found a %s namespace!" % sscha.Relax.__RELAX_NAMESPACE__)
        print ("Performing a whole relaxation.")
        
        relax.setup_from_namelist(namelist)
        
        # Save the final dynamical matrix
        print ("Saving the final dynamical matrix in %s" % (relax.minim.dyn_path + "_popuation%d_" % relax.minim.population))
        relax.minim.dyn.save_qe(relax.minim.dyn_path + "_population%d_" % minim.population)
        
        # Plot the results
        relax.minim.plot_results(args.save_data, args.plot)
        
        exit(0)


    # Parse the input file
    t1 = time.time()
    print ("Loading everything...")
    minim.setup_from_namelist(namelist)
    t2 = time.time()
    print ("Loaded in %.4f seconds." % (t2 - t1))

    # Start the minimization
    print ()
    print ("Initialization...")
    minim.init()
    print ("System initialized.")
    minim.print_info()
    print ()

    # If present, setup the custom functions
    if sscha.Utilities.__UTILS_NAMESPACE__ in namelist:
        cfs = sscha.Utilities.get_custom_functions_from_namelist(namelist, minim.dyn)    
        minim.run(custom_function_pre = cfs[0], 
                  custom_function_gradient = cfs[1],
                  custom_function_post = cfs[2])
    else:
        minim.run()
        
    print ()
    print ("Minimization ended.")
    minim.finalize(2)

    # Save the final dynamical matrix
    print ("Saving the final dynamical matrix in %s" % ('final_sscha_dyn_pop%d' % minim.population))
    print ('Saving the final centroids into {}'.format('final_sscha_structure_pop%d.scf' % minim.population))
    print ()
    minim.dyn.save_qe('final_sscha_dyn_pop%d' % minim.population)
    minim.dyn.structure.save_scf('final_sscha_structure_pop%d.scf' % minim.population)

    # Plot the results
    minim.plot_results(args.save_data, args.plot)
    return 0
