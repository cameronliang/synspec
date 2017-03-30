import numpy as np
import matplotlib.pyplot as pl
import roman
import re


from Model import WavelengthArray,get_transitions_params,simple_spec,bParameter
from IonizationModel import DefineIonizationModel,ion_list


def convert_ion_name(ion_name):
    match = re.match(r"([a-zA-Z]+)([0-9]+)", ion_name, re.I)
    atom = match.groups()[0].title()
    state = roman.toRoman(int(match.groups()[1]))
    return atom,state

def SingleValue_Spec(ion_model,wavelength_array,alpha,bNT,z,size,atom=None,state=None):
    """
    Compute the spectrum for one set of physical 
    properties 
    
    Parameters
    ----------
    ion_model: obj
        Class object defined by model name and redshift
    wavelength_array: array
    alpha: array_like
        Physical parameters with order: (lognH,logZ,logT)
    bNT: float
        Non-thermal velocity dispersion [km/s]
    z: float
        Redshift of the absorption line
    size: float
        thickness of cloud               [kpc]
    atom: str
        Name of the atom, e.g., 'C'
    state: str
        Ionization state in roman numeral, e.g, 'II'   
    
    Returns:
    ----------
    spec: array
        Normalized flux
    """

    ion_names,atomic_params = get_transitions_params(min(wavelength_array),max(wavelength_array),z,atom,state)

    spec = []
    spec = np.ones((len(ion_names),len(wavelength_array)))
    for i, ion in enumerate(ion_names):
        if ion in ion_list(): 
            atom,state = convert_ion_name(ion)
            ion_mass = atomic_params[i][-1]
            logN = ion_model.columndensity(alpha,ion,size)
            b = bParameter(alpha[2], bNT,ion_mass)
            spec[i] = simple_spec(logN, b, z, wavelength_array, atom=atom, state=state, lsf=1)
            
    spec = np.product(spec,axis=0)
    return spec 


def ProduceSpec_alongLOS(ion_model,wavelength_array,lognHs,logZs,logTs,bNTs,zs,sizes,
                        atom=None,state=None):
    
    alphas = np.array([lognHs,logZs,logTs]).T

    if isinstance(zs,float):
        zs = np.ones(len(lognHs)) * zs
    if isinstance(sizes,float):
        sizes = np.ones(len(lognHs)) * sizes
    if isinstance(bNTs,float):
        bNTs = np.ones(len(lognHs)) * bNTs

    spec = np.ones((len(alphas),len(wavelength_array)))
    for i, alpha in enumerate(alphas):
        spec[i] = SingleValue_Spec(ion_model,wavelength_array,alpha,bNTs[i],zs[i],sizes[i],atom,state)
    
    spec = np.product(spec,axis=0)
    return spec

def single_cloud_interactive():
    """Example spectrum from one cloud"""

    wave_start = float(raw_input('Start wavelength: '))
    wave_end   = float(raw_input('Stop wavelength: '))
    dv         = float(raw_input('resolution/pixel dv[km/s]: '))

    lognH     = float(raw_input('lognH[cm-2]: '))
    logZ      = float(raw_input('logZ[Zsun]: '))
    logT     = float(raw_input('logT[K]: '))
    bNT       = float(raw_input('Nonthernal dispersion[km/s]: '))
    z         = float(raw_input('Redshift: '))
    size      = float(raw_input('Cloud thickness[kpc]: '))

    atom       = raw_input('Atom: ')
    state      = raw_input('State: ')

    if atom == '' or state == '':
        atom = None; state = None

    # Define input wavenelgth array 
    wavelength_array = WavelengthArray(wave_start,wave_end,dv)
    
    # Define input parameters
    alpha = np.array([lognH,logZ,logT])
    
    # Define model
    model_name='photo_collision_thin'; 
    ion_model = DefineIonizationModel(model_name,z)

    # If only want a specfic transition
    spec = SingleValue_Spec(ion_model,wavelength_array,alpha,bNT,z,size,atom,state)

    # Plot spectrum
    pl.step(wavelength_array,spec)
    pl.ylim([0,1.4])
    pl.show()


def single_cloud_example():
    """Example spectrum from one cloud"""

    # Define input wavenelgth array 
    wavelength_array = WavelengthArray(1000,1560,7.5)
    
    # Define input parameters
    lognH = -3; logZ = 0; logT = 4.2
    alpha = np.array([lognH,logZ,logT])
    bNT    = 29.0
    z    = 0.0    
    size = 1.0 # kpc
    
    # Define model
    model_name='photo_collision_thin'; 
    ion_model = DefineIonizationModel(model_name,z)

    # If only want a specfic transition
    spec = SingleValue_Spec(ion_model,wavelength_array,alpha,bNT,z,size,'C','II')

    # If user want all of the transitions within the wavelength region
    #spec = SingleValue_Spec(ion_model,wavelength_array,alpha,bNT,z,size)    

    # Plot spectrum
    pl.step(wavelength_array,spec)
    pl.ylim([0,1.4])
    pl.show()

def MutipleClouds_LOS_example():
    """Example spectrum from a LOS of multiple clouds"""
    
    # Define input wavenelgth array 
    wavelength_array = WavelengthArray(1000,1560,7.5)

    # All physical parameters become arrays
    lognHs = [-4,-4.1,-3.6]
    logZs  = [0, 0, 0]
    logTs   = [4.3,4.2,4.0]

    # These parameters can also be array, same length as physical properties.
    bNT    = 29.0
    z      = 0.0    
    size   = 1.0 # kpc

    # Define model
    model_name='photo_collision_thin'; 
    ion_model = DefineIonizationModel(model_name,z)

    # Compute spectrum
    spec = ProduceSpec_alongLOS(ion_model,wavelength_array,lognHs,logZs,logTs,bNT,z,size)

    # Plot spectrum
    pl.step(wavelength_array,spec)
    pl.ylim([0,1.4])
    pl.show()

if __name__ == '__main__':
    
    #single_cloud_interactive()
    #single_cloud_example()
    MutipleClouds_LOS_example()
    

