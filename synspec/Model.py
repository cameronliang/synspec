################################################################################
#
# Model.py   		(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Voigt profile model; Produce prediction of model given the config file   
# (specifically the initial parameters). It can be imported to produce 
# a voigt profile on demand specified by user as well. 
################################################################################

import numpy as np
from scipy.special import wofz
import roman
import sys
import os
import re

from IonizationModel import DefineIonizationModel

# constants [cgs units]
h  = 6.6260755e-27   # planck constant
kB = 1.380658e-16    # Boltzmann constant
c  = 2.99792458e10   # speed of light
m  = 9.10938291e-28  # electron mass
mH = 1.67e-24        # proton mass
e  = 4.80320425e-10  # electron charge
sigma0 = 0.0263      # Cross section [cm^2/sec]

# Units Conversion
cm_km = 1.e-5  # Convert cm to km
km_cm = 1.e5   # Convert km to cm
ang_cm = 1.e-8 # Convert Angstrom to cm
kpc_cm = 3.0856e21 # Convert kpc to cm 


def convolve_lsf(flux,lsf):
	if len(flux) < len(np.atleast_1d(lsf)):
		# Add padding to make sure to return the same length in flux.
		padding = np.ones(len(lsf)-len(flux)+1)
		flux = np.hstack([padding,flux])
    	
		conv_flux = 1-np.convolve(1-flux,lsf,mode='same') /np.sum(lsf)
		return conv_flux[len(padding):]

	else:
		# convolve 1-flux to remove edge effects wihtout using padding
		return 1-np.convolve(1-flux,lsf,mode='same') /np.sum(lsf)

def convert_ion_notation(atom,state):
    if re.search('\*',state):
        state = state.replace('*','')
    return atom.lower()+str(roman.fromRoman(state))

def GetIonLists(atoms,states):
    ions_lists = []
    for i in xrange(len(atoms)):
        ion = convert_ion_notation(atoms[i],states[i])
        ions_lists.append(ion)
    ions_lists = list(set(ions_lists))
    return np.array(ions_lists)

def get_transitions_params(wave_start,wave_end,redshift,atom=None,state=None):
    """
    Extract the ionic and tranisiton properties based on the atom and 
    the state, given the redshift and observed wavelength regime.

    Parameters:
    ----------
    atom: str
        names of atoms of interests (e.g., 'H', 'C','Si')
    state: str
        ionization state of the atom in roman numerals (e.g., 'I', 'II')
    wave_start: float
        the minimum observed wavelength 
    wave_end: float
        the maximum observed wavelength
    redshift: float
        redshift of the absorption line

    Return:
    ----------
    transitions_params_array: array_like
        [oscilator strength, rest wavelength, dammping coefficient, mass of the atom] for all transitions within the wavelength interval. shape = (number of transitions, 4) 
    """
    amu = 1.66053892e-24   # 1 atomic mass in grams


    # Absolute path
    data_path = os.path.dirname(os.path.abspath(__file__)) 
    data_file = data_path + '/data/atom.dat'
    atoms,states  = np.loadtxt(data_file, dtype=str,unpack=True,usecols=[0,1])
    wave,osc_f,Gamma,mass = np.loadtxt(data_file,unpack=True,usecols=[2,3,4,5])
    mass = mass*amu 

    if atom is None and state is None:
        inds = np.where((wave >= wave_start/(1+redshift)) & 
                        (wave < wave_end/(1+redshift)))[0]
    else:
        inds = np.where((atoms == atom) & 
                        (states == state) & 
                        (wave >= wave_start/(1+redshift)) & 
                        (wave < wave_end/(1+redshift)))[0]

    if len(inds) == 0:
        return [], np.empty(4)*np.nan
    else:
        ions_lists = GetIonLists(atoms[inds],states[inds])
        atomic_params = np.array([osc_f[inds],wave[inds],Gamma[inds], mass[inds]]).T
        return ions_lists,atomic_params


def WavelengthArray(wave_start, wave_end, dv):
    """
    Create Wavelength array with resolution dv
    
    Parameters:
    ----------
    wave_start: float
        starting wavelength of array [\AA]
    wave_end: float
        ending wavelength of array [\AA]
    dv: float
        resolution element [km/s]
    
    Returns
    ----------
    wavelength_array: array_like
    """ 
    
    c = 299792.458       # speed of light  [km/s]
    wave_start = float(wave_start); wave_end = float(wave_end)

    # Calcualte Total number of pixel given the resultion and bounds of spectrum
    total_number_pixel = np.int(np.log10(wave_end/wave_start) / np.log10(1 + dv/c) + 0.5)
    array_index        = np.arange(0,total_number_pixel,1)
    
    # Return wavelength array
    return wave_start * ((1 + dv/c)**array_index)

def voigt(x, a):
    """
    Real part of Faddeeva function, where    
    w(z) = exp(-z^2) erfc(jz)
    """
    z = x + 1j*a
    return wofz(z).real

def Voigt(b, z, nu, nu0, Gamma):
    """
    Generate Voigt Profile for a given transition

    Parameters:
    ----------
    b: float
        b parameter of the voigt profile
    z: float
        resfhit of the absorption line
    nu: array_like
        rest frame frequncy array
    nu0: float
        rest frame frequency of transition [1/s]
    Gamma: float
        Damping coefficient (transition specific)

    Returns:
    ----------
    V: array_like
        voigt profile as a function of frequency
    """

    delta_nu = nu - nu0 / (1+z)
    delta_nuD = b * nu / c
    
    prefactor = 1.0 / ((np.pi**0.5)*delta_nuD)
    x = delta_nu/delta_nuD
    a = Gamma/(4*np.pi*delta_nuD)

    return prefactor * voigt(x,a)  

def bParameter(logT, b_nt, mass):
    """
    Combined thermal and non-thermal velocity

    Parameters:
    ----------
    mass: float
        mass of ion in the transition [grams]
    logT: array_like
        log10 of Temperature [Kelvin]
    b_nt: array_like
        non-thermal velocity dispersion [km/s]

    Returns:
    ----------
    b parameter: array_like; [km/s]
    """
    
    temp = 10**logT
    b_thermal = np.sqrt(2. * kB*temp / mass)*cm_km  # [km/s]
    return np.sqrt(b_thermal**2 + b_nt**2)


def General_Intensity(logN, b, z, wave, atomic_params):
    """
    Takes a general combination of atomic 
    parameters, without specifying the name of the transition 
    to compute the flux
    
    Parameters:
    ----------
    logN: float
        log10 of column density [cm-2] 
    b: float
        total b parameter [km/s]
    z: float
        redshift of system
    wave: 1D array
        observed wavelength array
    atomic_params: array
        array of oscillator strength, rest frame wavelength [\AA], damping coefficient, mass [grams] of the transition

    Returns:
    ----------
    intensity: array
        normalized flux of the spectrum    
    """
    f,wave0,gamma,mass = atomic_params

    # Convert to cgs units
    b       = b * km_cm       # Convert km/s to cm/s
    N       = 10**logN        # Column densiy in linear space
    lambda0 = wave0*ang_cm    # Convert Angstrom to cm
    nu0     = c/lambda0       # Rest frame frequency 
    nu      = c/(wave*ang_cm) # Frequency array 

    # Compute Optical depth
    tau = N*sigma0*f*Voigt(b,z,nu,nu0,gamma)

    # Return Normalized intensity
    return np.exp(-tau.astype(np.float))

def simple_spec(logN, b, z, wave, atom=None, state=None, lsf=1):
    """
    Generate a single component absorption for all transitions
    within the given observed wavelength, redshift for the desired
    atom and state

    Parameters:
    ----------
    logN: float
        log10 of column density [cm-2] 
    b: float
        total b parameter [km/s]
    z: float
        redshift of system
    wave: 1D array
        observed wavelength array

    Returns:
    ----------
    intensity: array
        normalized flux of the spectrum    
    """

    ion_lists,atomic_params = get_transitions_params(min(wave),max(wave),z,atom,state)
    
    spec = []
    n_transitions = len(atomic_params)
    for l in xrange(n_transitions):
        if not np.isnan(atomic_params[l]).any():
            model_flux = General_Intensity(logN,b,z,wave,atomic_params[l])
            spec.append(convolve_lsf(model_flux,lsf)) 
        else:
            return np.ones(wave)
    # Return the convolved model flux with LSF
    return np.product(spec,axis=0)


################################################################################