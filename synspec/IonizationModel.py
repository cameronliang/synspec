################################################################################
#
# IonizationModel.py   	(c) Cameron Liang 
#						University of Chicago
#     				    jwliang@oddjob.uchicago.edu
#
# Ionization model class for producing number or column densities of ions
################################################################################

import numpy as np
import os,re
import roman
from scipy.interpolate import RectBivariateSpline, interp1d,\
							  RegularGridInterpolator

from read_solarabund import SpecieMetalFraction, MetalFraction,\
							NumberFraction

################################################################################
# Utils
################################################################################


def convert_ion_name(ion_name):
    match = re.match(r"([a-zA-Z]+)([0-9]+)", ion_name, re.I)
    atom = match.groups()[0].title()
    state = roman.toRoman(int(match.groups()[1]))
    return atom,state


def ion_list():
	"""List of ions with pre-computed CLOUDY ionization fraction"""
	ions = np.array(['al2','c2','c3','c4','fe2','h1','mg2',
					 'n1','n2','n3','n4','n5','ne8','o1','o6',
					 'o7','o8','si2','si3','si4'])	
	return ions 

def Cloudy_InputParamers():
	lognH = np.arange(-6,0.2,0.2)
	logNHI = np.arange(15,19.2,0.2)
	logT = np.arange(4.0,7.2,0.2)
	return lognH, logNHI,logT
	
def Cloudy_InputParamers_redshift(): 
	low_redshift = np.arange(0,3.0,0.2)
	high_redshift = np.arange(3.0,7.5,0.5)
	redshift = np.sort(np.concatenate((low_redshift,high_redshift)))

	lognH = np.arange(-7.0, 0.2, 0.2)
	logT  = np.arange(3.5,7.1,0.2)
	return lognH,logT, redshift


################################################################################
# Models
################################################################################

def GenericModelInterp(modelz,ion_name,model_choice):
	"""
	Only photo_collision_thin model is available for the moment
	"""

	code_path = os.path.dirname(os.path.abspath(__file__)) 
	if model_choice == 'photo_collision_thin' or model_choice == 'photo_fix_logT_thin':
	
		# load the CLOUDY input parameters
		clognH,clogT,redshift = Cloudy_InputParamers_redshift()
		
		# Load the ionization fraction grid
		path = code_path + '/data/photo_collision_thin/CombinedGrid/cubes/'
		ind = int(np.where(abs(redshift-modelz) < 0.1)[0]) # Use the closest z
		ion = np.load(path + ion_name + '.npy')[ind,:,:]
		
		# Interpolate the function
		f = np.vectorize(RectBivariateSpline(clognH,clogT,ion))

	elif model_choice == 'photo_collision_noUVB':
		clognH,clogNHI,clogT = Cloudy_InputParamers()
		path = code_path + '/data/photo_collision_rahmati/f0.0/cubes/'
		ion = np.load(path + ion_name + '.npy')
		f = np.vectorize(RectBivariateSpline(clognH,clogT,ion))

	elif model_choice == 'photo_collision_rahmati':
		#will change name from optically_thick_rahmati to photo_collision_rahmati after the models are finished 
		clognH,clogNHI,clogT = Cloudy_InputParamers()
		path = code_path + '/' + model_choice + '/CombinedGrid/cubes/'
		cgamma_ratios = np.load(path + '/uvb_fraction.npy')
		ion = np.load(path + ion_name + '.npy')
		f = RegularGridInterpolator((clognH,clogT,cgamma_ratios),ion)

	elif model_choice == 'photo_collision_thick':
		clognH  = np.arange(-6,0.2,0.2)
		clogNHI = np.arange(15,19.2,0.2)
		clogT   = np.arange(3.8,6.2,0.2)
		path = code_path + '/' + model_choice + '/CombinedGrid/cubes/'
		ion =  np.load(path + ion_name + '.npy') 
		f = RegularGridInterpolator((clognH,clogNHI,clogT),ion)

	return f

def GetAllIonFunctions(modelz,model_choice):
	ions_names = ion_list()
	f = [] 
	for ion_name in ions_names: 
		f.append(GenericModelInterp(modelz,ion_name,model_choice)) 
	f = np.array(f)
	
	# Make the dictionary between functions and ionization state. 
	dict_intepfunc = {}
	for i in range(len(ions_names)): dict_intepfunc[ions_names[i]] = f[i]
	return dict_intepfunc
	
################################################################################
# Physics related Utils 
################################################################################

def ComputeGammaRatio(lognH):
	"""
	ratio = Gamma/Gamma_UVB; 
	eqn 14. from Rahmati 2013.
	"""
	nH_ssh = 5.1*1e-4; # value taken from table 2 Rahmati+ 2013
	
	nH = 10**lognH
	ratio = 0.98*(1+(nH/nH_ssh)**1.64)**-2.28 + 0.02*(1+nH/nH_ssh)**-0.84
	return ratio

def logZfrac(logZ,specie):
	#logZ is in solar units already  
	logNx_NH = NumberFraction(specie) # number density ratio in the sun 
	return logZ + logNx_NH


################################################################################

class DefineIonizationModel:
	"""
	A Ionization model class 
	"""

	def __init__(self,model,model_redshift):
		self.model = model
		self.model_redshift = model_redshift
		self.logf_ion = GetAllIonFunctions(self.model_redshift,self.model) 

	def model_prediction(self,alpha,ion_name):
		"""
		Calculate number density given a specific ion, and the model 
		parameters in a photo-ionization model
		"""
		
		specie, _ = convert_ion_name(ion_name)

		lognH,logZ,logT = alpha
		if self.model == 'photo_collision_thin':		
			if -6 < lognH < 0 and 3.8 <= logT < 7:
				logn_ion = (self.logf_ion[ion_name](lognH,logT) + logZfrac(logZ,specie) + lognH)[0][0]
			else:
				logn_ion = -np.inf

		elif self.model == 'photo_collision_noUVB':
			logn_ion = (self.logf_ion[ion_name](lognH,logT) + logZfrac(logZ,specie) + lognH)[0][0]
		
		elif self.model == 'photo_collision_rahmati':
			if lognH < -6: 
				lognH = -6.0 # this is because the models were not run below -6. 
			elif lognH > 0:
				lognH = 0.
			
			gamma_ratio = ComputeGammaRatio(lognH)
			logn_ion = (self.logf_ion[ion_name]((lognH,logT,gamma_ratio)) + logZfrac(logZ,specie) + lognH)[0][0]

		return logn_ion

	def columndensity(self,alpha,ion_name,size):
		"""
		Calculate the column density given the ion name 
		and cell size after the model is defined. 

		Parameters: 
		-----------
		ion_name: str
			Name of the ion, e.g 'h1','c3','s2'

		size: float
			Thickness of a uniform density cloud [kpc]

		"""
		kpc_cm = 3.0856e21 # Convert kpc to cm 
		log10_cellsize = np.log10(size * kpc_cm)
		logN = self.model_prediction(alpha,ion_name)+log10_cellsize
		
		return logN


if __name__ == '__main__':

	# usage
	import sys
	ion_model = DefineIonizationModel('photo_collision_thin',0.0)
	alpha = np.array([-3,0,4.2])
	print ion_model.model_prediction(alpha,'c2')
	print ion_model.model_prediction(alpha,'c3')
	print ion_model.model_prediction(alpha,'c4')
	print ion_model.model_prediction(alpha,'s2')
	print ion_model.model_prediction(alpha,'s3')
	print ion_model.model_prediction(alpha,'s4')
	print ion_model.model_prediction(alpha,'o1')
	print ion_model.model_prediction(alpha,'o6')
	print ion_model.model_prediction(alpha,'n5')
