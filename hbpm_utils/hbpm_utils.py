#!/usr/bin/env python
# coding: utf-8

'''
This file contains the various required functions for the Hierarchical Post-Merger Bayesian analysis (HBPM). 
'''

import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
import scipy.stats as st
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde as kde
from scipy.stats.mstats import mquantiles as mq
from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde
from glob import glob
import pandas as pd
import sys 
import os
from tabulate import tabulate
import pickle
from astropy import cosmology as co
from astropy import units as u
from tqdm import tqdm


'''
Table of Contents

Section 1: Classes
Section 2: String and filename parsing
Section 3: Calculations with physical quantities
Section 4: Data loading and priors
Section 5: Likelihoods
Section 6: Posteriors
Section 7: Post-processing, Plotting, and Saving
Section 8: Plotting BayesWave Results

'''


## SECTION 1: CLASSES

## faster wrappers for the fpeak prior and posterior KDEs
## issue is that for N_samples >> 1, calling the resulting KDE is impractically slow
## soln: create prior object that holds the samples and actual KDE, but evaluates the prior probability
##       by doing fast 1D interpolation
## note re: PEsummary - need to install from github, conda installation doesn't have bounded_1d_kde
class Prior_f:
    '''
    fpeak prior class
    
    Initialize by feeding samples, a frequency grid, and KDE parameters (boundary type and bandwidth).
    samples (array)              : array of fpeak prior samples
    fmin, fmax (float)           : Observed range (kHz). Used as boundaries for bounded kde and defines valid interpolation range
    fgrid (float)                : Number of points in frequency used to compute the linear interpolator.
    boundary (str)               : KDE boundary type. Can be 'Standard' (usual scipy Gaussian KDE, no boundary conditions)
                                   or any of the PEsummary bounded KDE options ('Reflection' and 'Transform')
    kde_bandwidth (float or str) : bw_method value passed to scipy KDE
    
    Attributes
    samples    : original fpeak prior samples, kept for convenience
    fmin, fmax : Valid interpolation range. Also kde bounds.
    kde        : Kernel density estimator
    pdf        : Interpolator for prior density
    '''
    def __init__(self,samples,boundary,fmin=1.5,fmax=4,fgrid=200,kde_bandwidth=0.15):
        self.samples = samples
        self.fmin = fmin
        self.fmax = fmax
        self.boundary = boundary
        self.bw_method = kde_bandwidth
        if len(samples.shape) > 1:
            self.ndet = samples.shape[1]
            self.samples = samples
        else:
            self.ndet = 1
            self.samples = samples.reshape(-1,1)
        fs = np.linspace(self.fmin,self.fmax,fgrid)
        kdes = []
        for i in range(self.ndet):
            samples_i = self.samples[:,i]
            if boundary=='Standard':
                ## default scipy kde
                kdes.append(kde(samples_i,bw_method=kde_bandwidth))
            else:
                kdes.append(bounded_1d_kde(samples_i,xlow=self.fmin,xhigh=self.fmax,method=boundary,bw_method=kde_bandwidth))
            if i==0:
                fdist = kdes[0](fs)
            else:
                fdist = fdist*kdes[i](fs)
        
        self.kde = kdes
        self.pdf = interp1d(fs,fdist)

    
    def rvs(self,size=1,rng=None,det_idx=None):
        '''
        Arguments:
            size (int) : number of samples to draw
            rng (numpy rng) : generator object to use
            det_idx (int) : If prior is multi-detector, use this to specify which column you want to draw from. If no specification is made, the array will be flattened.
        '''
        if det_idx is not None:
            samples = self.samples[:,det_idx]
        else:
            samples = self.samples.flatten()
        if rng is not None:
            return rng.choice(samples,size=size)
        else:
            return np.random.choice(samples,size=size)

class Posterior_f:
    '''
    fpeak posterior class. Similar to prior class, primary purpose is to speed up the KDE calls
    
    Initialize by feeding samples, a frequency grid, and KDE parameters (boundary type and bandwidth).
    samples (array) : array of fpeak prior samples
    fmin, fmax      : Observed range (kHz). Used as boundaries for bounded kde and defines valid interpolation range
    fgrid (float)   : Number of points in frequency used to compute the linear interpolator.
    boundary (str)  : KDE boundary type. Can be 'Standard' (usual scipy Gaussian KDE, no boundary conditions)
                      or any of the PEsummary bounded KDE options ('Reflection' and 'Transform')
    kde_bandwidth   : bw_method value passed to scipy KDE
    
    Attributes
    samples    : Original fpeak posterior samples, kept for convenience
    fmin, fmax : Valid interpolation range. Also kde bounds.
    kde        : Kernel density estimator (or list thereof)
    pdf        : Interpolator for posterior density
    ndet       : Number of detectors
    '''
    def __init__(self,samples,boundary,fmin=1.5,fmax=4,fgrid=200,kde_bandwidth=0.15):
        self.fmin = fmin
        self.fmax = fmax
        self.boundary = boundary
        self.bw_method = kde_bandwidth
        ## check to see if samples are separated by detector
        if len(samples.shape) > 1:
            self.ndet = samples.shape[1]
            self.samples = samples
        else:
            self.ndet = 1
            self.samples = samples.reshape(-1,1)
        fs = np.linspace(self.fmin,self.fmax,fgrid)
        kdes = []
        for i in range(self.ndet):
            samples_i = self.samples[:,i]
            if boundary=='Standard':
                ## default scipy kde
                kdes.append(kde(samples_i,bw_method=kde_bandwidth))
            else:
                kdes.append(bounded_1d_kde(samples_i,xlow=self.fmin,xhigh=self.fmax,method=boundary,bw_method=kde_bandwidth))
            if i==0:
                fdist = kdes[0](fs)
            else:
                fdist = fdist*kdes[i](fs)
        
        self.kde = kdes
        self.pdf = interp1d(fs,fdist)

class Posterior_M:
    '''
    Mchirp posterior class. Similar to fpeak posterior class, primary purpose is to speed up the KDE calls
    
    Initialize by feeding samples, a mass grid, and KDE parameters (boundary type and bandwidth).
    samples (array) : array of Mchirp prior samples
    Mmin, Mmax      : Mass range (solar masses). Used as boundaries for bounded kde and defines valid interpolation range
    Mgrid (int)   : Number of points in frequency used to compute the linear interpolator.
    boundary (str)  : KDE boundary type. Can be 'Standard' (usual scipy Gaussian KDE, no boundary conditions)
                      or any of the PEsummary bounded KDE options ('Reflection' and 'Transform')
    bw_method   : bw_method value passed to scipy KDE
    
    Attributes
    samples    : original fpeak posterior samples, kept for convenience
    Mmin, Mmax : Valid interpolation range. Also kde bounds.
    kde        : Kernel density estimator
    pdf        : Interpolator for prior density
    '''
    def __init__(self,samples,boundary,Mmin=0,Mmax=5,Mgrid=25000,bw_method=None):
        self.samples = samples
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.boundary = boundary
        self.bw_method = bw_method
        if boundary=='Standard':
            ## default scipy kde
            self.kde = kde(samples,bw_method=bw_method)
        else:
            self.kde = bounded_1d_kde(samples,xlow=self.Mmin,xhigh=self.Mmax,method=boundary,bw_method=bw_method)
        Ms = np.linspace(self.Mmin,self.Mmax,Mgrid)
        self.pdf = interp1d(Ms,self.kde(Ms))
        
## SECTION 2: STRING/NAME PARSING

@np.vectorize
def parse_datafile(data_file):
    '''
    Function to parse the Soultanis simulation datafile names to return the mass string notation used in the BayesWave
    injection infrastructure. (e.g. sfhx_mtot2times1.0400_q0.8000_id2poles.dat -> '092116', because m1=0.92Mo & m2=1.16Mo.)
    
    Input:
        data_file (str) : '/path/to/Soultanis/simulation/file.dat' (or 'file_name.dat', either works)
    
    Returns:
        mstr (str) : Six-digit mass string as described above
    '''
    
    if "mtot2times" in data_file:
        wavename = data_file.split('/')[-1]
        wavename = wavename.replace('mtot2times','')
        wavename = wavename.replace('q','')
        wavename = wavename.replace('_id2poles.dat','')
        # Now walk backwards through the waveform label.  This makes it
        # easier to handle EOS names with underscores...
        # q
        q = float(wavename.split('_')[-1])
        # mass
        half_mtot = wavename.split('_')[-2]
        mtot = 2*float(half_mtot)
        m1 = "{:0.2f}".format(mtot/(1 + 1/q))
        m2 = "{:0.2f}".format(mtot/(1 + q))
        
    elif "mttot2times" in data_file:
        wavename = data_file.split('/')[-1]
        wavename = wavename.replace('sly-','')
        wavename = wavename.replace('dd2-','')
        wavename = wavename.replace('mttot2times','')
        wavename = wavename.replace('q','')
        wavename = wavename.replace('_id2poles.dat','')
        # Now walk backwards through the waveform label.  This makes it
        # easier to handle EOS names with underscores...
        # q
        q = float(wavename.split('_')[-1])
        # mass
        half_mtot = wavename.split('_')[-2]
        mtot = 2*float(half_mtot)
        m1 = "{:0.2f}".format(mtot/(1 + 1/q))
        m2 = "{:0.2f}".format(mtot/(1 + q))

    elif ("m1-" in data_file) and ("m2-" in data_file):
        wavename = data_file.split('/')[-1]
        wavename = wavename.replace('sly-','')
        wavename = wavename.replace('dd2-','')
        wavename = wavename.replace('m1-','')
        wavename = wavename.replace('m2-','')
        wavename = wavename.replace('_id2poles.dat','')

        # q
        q = 1.00

        # mass
        m1 = "{:0.2f}".format(float(wavename.split('_')[-1]))
        m2 = "{:0.2f}".format(float(wavename.split('_')[-2]))
    
    else:
        raise Exception('Input needs to be a filename in the standard Soultanis format')
    
    return (m1+m2).replace('.','')

@np.vectorize
def parse_mstr(mstr):
    '''
    Function to parse the 6-, 7-, or 8-digit mass strings used in the BayesWave injection infrastructure into individual masses.
    
    Input:
        mstr (str) : Mass string as described in parse_datafile(). Can be 6, 7, or 8 digits for now.
    
    Returns:
        m1, m2 (float) : Binary masses given in solar masses. m1 is the smaller of the pair.
    '''
    if len(mstr) == 7:
        m1 = float(mstr[:3])*1e-2
        m2 = float(mstr[3:])*1e-3
    elif len(mstr) == 8:
        m1 = float(mstr[:4])*1e-3
        m2 = float(mstr[4:])*1e-3
    elif len(mstr) == 6:
        m1 = float(mstr[:3])*1e-2
        m2 = float(mstr[3:])*1e-2
    else:
        raise Exception('Can only parse mass strings 6, 7, or 8 digits long.')
    return m1, m2

@np.vectorize
def parse_qm1m2(data_file):
    '''
    Function to get the mass ratio q and binary masses m1 and m2 from a Soultanis NR simulation filename.
    
    Arguments:
        data_file (str) : '/path/to/soultanis/sim/file.dat' (or 'file_name.dat', either works)
    
    Returns:
        q (float)      : Mass ratio m1/m2
        m1, m2 (float) : Binary masses in solar masses
    '''
    if "mtot2times" in data_file:
        wavename = data_file.split('/')[-1]
        wavename = wavename.replace('mtot2times','')
        wavename = wavename.replace('q','')
        wavename = wavename.replace('_id2poles.dat','')
        # Now walk backwards through the waveform label.  This makes it
        # easier to handle EOS names with underscores...
        # q
        q = float(wavename.split('_')[-1])
        # mass
        half_mtot = wavename.split('_')[-2]
        mtot = 2*float(half_mtot)
        m1 = mtot/(1 + 1/q)
        m2 = mtot/(1 + q)
    
    elif "mttot2times" in data_file:
        wavename = data_file.split('/')[-1]
        wavename = wavename.replace('sly-','')
        wavename = wavename.replace('dd2-','')
        wavename = wavename.replace('mttot2times','')
        wavename = wavename.replace('q','')
        wavename = wavename.replace('_id2poles.dat','')
        # Now walk backwards through the waveform label.  This makes it
        # easier to handle EOS names with underscores...
        # q
        q = float(wavename.split('_')[-1])
        # mass
        half_mtot = wavename.split('_')[-2]
        mtot = 2*float(half_mtot)
        m1 = mtot/(1 + 1/q)
        m2 = mtot/(1 + q)
    

    elif ("m1-" in data_file) and ("m2-" in data_file):
        wavename = data_file.split('/')[-1]
        wavename = wavename.replace('sly-','')
        wavename = wavename.replace('dd2-','')
        wavename = wavename.replace('m1-','')
        wavename = wavename.replace('m2-','')
        wavename = wavename.replace('_id2poles.dat','')

        # q
        q = 1.00

        # mass
        m1 = float(wavename.split('_')[-1])
        m2 = float(wavename.split('_')[-2])
    else:
        raise TypeError('Unknown file format.')
    return q, m1, m2


## SECTION 3: CALCULATIONS FOR PHYSICAL QUANTITIES

def calc_Mchirp(m1,m2):
    '''
    Function to calculate the binary chirp mass given binary masses m1 and m2.
    
    Inputs:
        m1, m2 (float) : Binary component masses in solar masses.
    
    Returns:
        Mchirp : Binary chirp mass
    '''
    return ((m1*m2)**(3/5))/(m1+m2)**(1/5)

def empirical_relation(fpeak,Mchirp):
    '''
    Function to calculate the empirical relation for R_1.6 in terms of fpeak and Mchirp.
    From table V of Vretinaris et al. (2020) http://arxiv.org/abs/1910.10856. Note that sigma_R16 = 0.117
    
    Inputs:
        fpeak (float)  : Post-merger peak frequency in kHz
        Mchirp (float) : Chirp mass in solar masses
    
    Returns:
        R16 (float) : Empirical relation prediction for R_1.6
    '''
    b0 = 43.796
    b1 = -19.984
    b2 = -12.921
    b3 = 4.674
    b4 = 3.371
    b5 = 1.26
    R16 = b0 + b1*Mchirp + b2*fpeak/Mchirp + b3*Mchirp**2 + b4*fpeak + b5*(fpeak/Mchirp)**2
    return R16


def empirical_relation_f(R16,Mchirp):
    '''
    Function to calculate the empirical relation for fpeak in terms of R_1.6 and Mchirp.
    From table IV of Vretinaris et al. (2020) http://arxiv.org/abs/1910.10856. Note that sigma_f = 0.053
    
    Inputs:
        R16 (float)  : R_1.6 in km
        Mchirp (float) : Chirp mass in solar masses
    
    Returns:
        fpeak (float) : Empirical relation prediction for the post-merger peak frequency
    '''
    b0 = 12.696
    b1 = -0.935
    b2 = -1.17
    b3 = 0.713
    b4 = -0.092
    b5 = 0.037
    fpeak = b0*Mchirp + b1*Mchirp**2 + b2*Mchirp*R16 + b3*Mchirp**3 + b4*R16*Mchirp**2 + b5*Mchirp*R16**2
    return fpeak

def empirical_relation_bootstrap(R16,Mchirp,bootstrap_coeffs):
    '''
    Function to calculate a bootstrapped distribution of empirical relation predictions for fpeak in terms of R_1.6 and Mchirp.
    Bootstrapped coefficients are in principle based on the relation from table IV of Vretinaris et al. (2020) 
        (see http://arxiv.org/abs/1910.10856.) This function only works if the bootstrapped relations are of the form
        f = Intercept + b0*M + b1*M^2 + b3*M^3 + b4*R*M^2 + b5*M*R^2
    
    Inputs:
        R16 (float)  : R_1.6 in km
        Mchirp (float) : Chirp mass in solar masses
        bootstrap_coeffs (Nx7 array of floats) : array of relation coefficients for N bootstrapped relations
    
    Returns:
        fpeak (Nx1 array of floats) : Distribution of empirical relation predictions for the post-merger peak frequency
    '''
    I = bootstrap_coeffs[:,0]#.reshape(-1,1).T
    b0 = bootstrap_coeffs[:,1]#.reshape(-1,1).T
    b1 = bootstrap_coeffs[:,2]#.reshape(-1,1).T
    b2 = bootstrap_coeffs[:,3]#.reshape(-1,1).T
    b3 = bootstrap_coeffs[:,4]#.reshape(-1,1).T
    b4 = bootstrap_coeffs[:,5]#.reshape(-1,1).T
    fpeaks = I + b0*Mchirp + b1*Mchirp**2 + b2*Mchirp*R16 + b3*R16*Mchirp**2 + b4*Mchirp*R16**2
    return fpeaks

def get_mchirp_range(Rtrue,fmin,fmax,suppress_error=False):
    '''
    Function to find the valid chirp mass range corresponding to a given EoS (as parameterized by R1.6). 
    "Valid" here means resulting in a peak frequency on the specified [fmin,fmax] range.
    
    Arguments:
        Rtrue (float) : True value of R1.6
        fmin (float) : Minimum peak frequency considered
        fmax (float) : Maximum peak frequency considered
        suppress_error (bool) : If True, returns NaNs when no valid chirp masses exist, instead of raising an error.
        
    Returns:
        Minimum and maximum of the valid chirp mass range.
    '''
    m_init = np.linspace(0.85,1.88,100)
    fs = np.linspace(fmin,fmax,50)
    invalid = []
    for i, m in enumerate(m_init):
        Rhat = empirical_relation(fs,m)
        if Rtrue < Rhat.min():
            invalid.append(m)
        if Rtrue > Rhat.max():
            invalid.append(m)
    if len(invalid)==len(m_init):
        if suppress_error==True:
            mlow = np.nan
            mhigh = np.nan
        else:
            raise ValueError('No valid chirp masses for this EoS.')
    else:
        m_good = m_init[[(m not in invalid) for m in m_init]]
        mlow, mhigh = m_good.min(), m_good.max()
    return mlow, mhigh


## SECTION 4: DATA LOADING AND PRIORS

def load_fprior(priordir,fmin=2000,fmax=4000):
    '''
    Function to load peak frequency prior samples from a BayesWave prior= run.
    
    Inputs:
        priordir (str) : '/path/to/directory/with/peak/frequency/prior/samples'
        fmin (int) : Minimum frequency used in the BayesWaveFpeak run (Hz)
        
    Returns:
        prior_H1L1 : Combined peak frequency prior samples from both H1 and L1 detectors
    '''
    prior_L1 = np.loadtxt(priordir+'/signal_recovered_whitened_waveform_fpeaks_min_'+str(fmin)+'_max_'+str(fmax)+'_L1.dat',skiprows=1)
    prior_H1 = np.loadtxt(priordir+'/signal_recovered_whitened_waveform_fpeaks_min_'+str(fmin)+'_max_'+str(fmax)+'_H1.dat',skiprows=1)
    prior_H1L1 = np.append(prior_L1[:,0],prior_H1[:,0])
    return prior_H1L1

def load_fsamples(datadir, fmin, fmax=5000, det='H1L1'):
    '''
    Load H1+L1 BayesWave recovery w/o prior.
    
    Arguments:
        datadir (str) : '/path/to/directory/with/BayesWave/Fpeak/output/'
        fmin (int) : minimum frequency used in BW recovery (usually 1500 or 2000 Hz)
        fmax (int) : maximum frequency used in BW recovery (usually 4000 or 5000 Hz)
        det (str) : detector ifos. Can be H1L1 to pool LIGO H1 and L1 samples, otherwise, should be the BayesWave suffix for the detector in question (e.g., H1, L1, V1)
    Returns:
        white_H1L1 (1D array) : combined Hanford and Livingston fpeak samples
    '''
    if det=='H1L1':
        white_L1 = np.genfromtxt(datadir+'/signal_recovered_whitened_waveform_fpeaks_min_'+str(fmin)+'_max_'+str(fmax)+'_L1.dat',
                          skip_header=1,usecols=0,filling_values=0)
        white_H1 = np.genfromtxt(datadir+'/signal_recovered_whitened_waveform_fpeaks_min_'+str(fmin)+'_max_'+str(fmax)+'_H1.dat',
                          skip_header=1,usecols=0,filling_values=0)
        white_H1L1 = np.append(white_L1,white_H1)
        return white_H1L1
    else:
        white_samp = np.genfromtxt(datadir+'/signal_recovered_whitened_waveform_fpeaks_min_'+str(fmin)+'_max_'+str(fmax)+'_'+det+'.dat',
                          skip_header=1,usecols=0,filling_values=0)
        return white_samp

def load_BayesWave_fprior(priordir,det,rng,draws='all',fmin=1000,fmax=5000):
    '''
    Function to load a BayesWave peak frequency prior.
    
    Arguments:
        priordir (str) : location of BayesWave fpeak prior output directory
        rng (Generator) : Numpy RNG instance (e.g. np.random.default_rng(seed))
        draws (str) or (int) : Can be 'all' or an integer. If all, the entire prior is loaded.
                                If an integer, draws is the total number of samples to draw from the prior.
        det (str) : Detector. Can be H1, L1, or H1L1.
        fmin (int) : minimum frequency used in the BayesWaveFpeak analysis.
    Returns:
        prior_draws (1D array) : peak frequency prior samples for one event
    '''
    if det=='H1L1':
        fprior_H1 = np.loadtxt(
            priordir+'/signal_recovered_whitened_waveform_fpeaks_min_'+str(fmin)+'_max_'+str(fmax)+'_'+'H1'+'.dat',
            skiprows=1)[:,0]
        fprior_L1 = np.loadtxt(
            priordir+'/signal_recovered_whitened_waveform_fpeaks_min_'+str(fmin)+'_max_'+str(fmax)+'_'+'L1'+'.dat',
            skiprows=1)[:,0]
        fprior = np.concatenate((fprior_H1,fprior_L1))
    else:
        fprior = np.loadtxt(
            priordir+'/signal_recovered_whitened_waveform_fpeaks_min_'+str(fmin)+'_max_'+str(fmax)+'_'+det+'.dat',
            skiprows=1)[:,0]
    if draws=='all':
        prior_draws = fprior
    elif type(draws)==int:
        if draws > len(fprior):
            print("Warning: More draws requested than number of provided samples. This will result in duplicate draws.")
        prior_draws = rng.choice(fprior,size=draws)
    else:
        raise TypeError("Invalid specification of draws. Should be either 'all' or a positive integer.")
    return prior_draws

def draw_from_BayesWave_fprior(fprior,rng,draws='all'):
    '''
    Function to draw from an already-loaded BayesWave peak frequency prior.
    
    NOTE: This function assumes the given prior is in kHz and adjusts back to Hz 
          so as to play well with the other loading functions. Adjust manually as needed.
    
    Arguments:
        fprior (array or Prior_f object) : BayesWave fpeak prior samples or Prior_f
        rng (Generator) : Numpy RNG instance (e.g. np.random.default_rng(seed))
        draws (str) or (int) : Can be 'all' or an integer. If all, the entire prior is loaded.
                                If an integer, draws is the total number of samples to draw from the prior.
        det (str) : Detector. Can be H1, L1, or H1L1.
        fmin (int) : minimum frequency used in the BayesWaveFpeak analysis.
    Returns:
        prior_draws (1D array) : peak frequency prior samples for one event
    '''
    if draws=='all':
        if hasattr(fprior,'samples'):
            prior_draws = fprior.samples * 1000
        else:
            prior_draws = fprior*1000
    elif type(draws)==int:
        if hasattr(fprior,'rvs') and hasattr(fprior,'samples'):
            N_tot = len(fprior.samples)
            prior_draws = fprior.rvs(size=draws,rng=rng)*1000
        else:
            N_tot = len(fprior)
            prior_draws = rng.choice(fprior,size=draws)*1000
        if draws > N_tot:
            print("Warning: More draws requested than number of provided samples. This will result in duplicate draws.")
    else:
        raise TypeError("Invalid specification of draws. Should be either 'all' or a positive integer.")
    return prior_draws

def load_BayesWave_aggregate_prior(prior_head,run_list,rng,det='H1L1'):
    '''
    Function to load an aggregate prior from many BayesWave prior runs.
    Intended for use with BayesWave runs over noise, but can in principle be used for other purposes.
    
    Arguments:
        prior_head (str) : '/path/to/directory/'; 
                           should contain all desired subdirectories, each with fpeak samples from a BayesWave prior run.
        run_list (list of str) : List of subdirectory names.
        rng (Generator) : Numpy RNG instance (e.g. np.random.default_rng(seed))
        det (str) : Detector. Can be H1, L1, or H1L1.
        fmin, fmax, fgrid (float) : Parameters used for the linear interpolator, e.g. np.linspace(fmin,fmax,fgrid).
                                    fmin and fmax should cover the prior sample range.
    Returns:
        agg_prior (array) : Aggregate prior samples.
    '''
    fs_prior = np.linspace(fmin,fmax,fgrid)
    agg_prior = np.array([])

    for run in run_list:
        prior_i = load_BayesWave_fprior(prior_head+run,'H1L1',np.random.default_rng(42))/1000
        agg_prior = np.append(agg_prior,prior_i)
    
    return agg_prior

def filter_fsamples(samples,fmin=1500,fmax=4000):
    '''
    Function to filter peak frequency samples to specified range.
    
    Arguments:
        samples (array) : Peak frequency samples.
        fmin (float) : Lower frequency cutoff.
        fmax (float) : Higher frequency cutoff.
    
    Returns:
        samples_filtered (array) : Filtered peak frequency samples.
    '''
    samples_filtered = samples[np.logical_and(samples>=fmin,samples<=fmax)]
    
    return samples_filtered

def load_BayesWave_fpeak(datadir,rng,ifos=['H1','L1'],nsamp=2500,bwfmin=1000,bwfmax=5000,fmin=1500,fmax=4000,
                         use_prior='array',prior=None,priordir=None):
    '''
    Function to load a BayesWave peak frequency recovery, using the BayesWave fpeak prior.
    
    Arguments:
        datadir (str) : location of BayesWave fpeak output directory
        rng (Generator) : Numpy RNG instance (e.g. np.random.default_rng(seed))
        ifos (list of str) : detector ifos. Can be any combination of H1, L1, and V1, e.g. ['H1','L1']
        nsamp (int) : Total number of samples in the fpeak recovery.
        bwfmin (int) : Minimum frequency (in Hz) of the *BayesWave* peak frequency analysis.
        bwfmax (int) : Maximum frequency (in Hz) of the *BayesWave* peak frequency analysis.
        fmin (int) : Minimum frequency (in Hz) considered in *this* analysis. Samples are filtered accordingly.
        fmax (int) : Maximum frequency (in Hz) considered in *this* analysis. Samples are filtered accordingly.
        use_prior (str) : Whether (and if so, how) to supplement the peak frequency with draws from the prior. 
                          Can be 'no', 'array', or 'load'. If 'array', need to specify prior=; 
                          if 'load', need to specify priordir=.
        prior (array) : Peak frequency prior samples. Recommended practice is the samples attribute of a Prior_f object.
                        Can also be a dictionary of arrays with desired ifos as above for keys.
        priordir (str) : location of BayesWave fpeak prior output directory if one would rather load prior samples
                         directly (not optimal but can be useful in some situations)
    Returns:
        fsamp (1D array) : peak frequency samples for one event
    '''
    ## make sure ifos is a list
    if type(ifos) is str:
        ifos = ifos.split(',')
    ## load
    for i, det in enumerate(ifos):
        fsamp_i = filter_fsamples(load_fsamples(datadir,bwfmin,fmax=bwfmax,det=det),fmin=fmin,fmax=fmax)
        ngood = len(fsamp_i)
        nnoise = nsamp - ngood
        ## draw from prior
        if use_prior != 'no':
            if use_prior=='array':
                if prior is None:
                    raise TypeError("If use_prior='array', must specify an array of fpeak prior samples.")
                if type(prior) is dict:
                    prior_i = prior[det]
                else:
                    prior_i = prior
                fprior = draw_from_BayesWave_fprior(prior_i,rng,draws=nnoise)
            elif use_prior=='load':
                if priordir is None:
                    raise TypeError("If use_prior='load', must specify an fpeak prior directry.")
                fprior = load_BayesWave_fprior(priordir,det,rng,draws=nnoise)
            elif use_prior=='uniform':
                fprior = st.uniform.rvs(loc=1500,scale=2500,size=nnoise,random_state=rng)
            else:
                raise TypeError("Invalid use_prior specification. use_prior can be 'no', 'array', 'load', or 'uniform'.")
            fsamp_i = np.concatenate((fsamp_i,fprior.flatten()))
        ## one column for each detector
        if i==0:
            fsamp = fsamp_i.reshape(-1,1)
        else:
            fsamp = np.hstack((fsamp, fsamp_i.reshape(-1,1)))

    return fsamp

def load_Bilby_Mchirp(datadir):
    '''
    Function to load a Bilby chirp mass recovery.
    
    Arguments:
        datadir (str) : location of Bilby Mchirp output directory
        
    Returns:
        Msamp (1D array) : chirp mass samples for one event
    '''
    bilby_post = pd.read_csv(datadir+'/posterior.txt',delimiter=' ')
    Msamp = bilby_post['chirp_mass'].to_numpy()
    return Msamp

def load_Rprior(Rprior_file,plot=True,return_samples=False):
    '''
    Function to load the prior for R1.6.
    Currently we are using the multimessenger prior of Dietrich et al. (2020) https://www.science.org/doi/10.1126/science.abb4317
    
    Arguments:
        Rprior_file (str) : '/path/to/R16/prior/samples.txt'
        plot (bool) : Whether to also return a plot of the R16 prior KDE.
        return_samples (bool) : Whether to also return the loaded R16 prior samples.
        
    Returns:
        Rs (array) : R1.6 range to consider (and range on which KDE is valid!)
        Rprior_kernel (scipy.stats.gaussian_kde) : R_1.6 prior KDE
        (Optional) Rprior_samples (array) : R_1.6 prior samples used to construct KDE. Only returned if return_samples==True
    '''
    ## load prior
    Rprior_samples = np.loadtxt(Rprior_file)
    ## set grid in R
    Rs = np.linspace(Rprior_samples.min(),Rprior_samples.max(),200)
    ## R16 prior KDE
    Rprior_kernel = kde(Rprior_samples)
    ## plot
    if plot==True:
        plt.figure()
        plt.title('$R_{1.6}$ Prior')
        plt.plot(Rs,Rprior_kernel.pdf(Rs))
        plt.xlabel('$R_{1.6}$ (km)')
        plt.ylabel('p(R)')
        plt.gca().set_yticks([])
        mplcyberpunk.add_glow_effects()
        plt.show()
    ## (optional) return KDE and samples
    if return_samples==True:
        return Rs, Rprior_kernel, Rprior_samples
    else:
        return Rs, Rprior_kernel

def load_bootstrap(boot_spec):
    '''
    Function to load the sampled empirical relation coefficients.
    
    Arguments:
        boot_spec (str): Can be 'default', in which case the standard set of samples used for Criswell+2022 will be used.
                         Otherwise can be '/path/to/file/with/samples.txt', but the format needs to be the same as the Criswell+22 version.
        
    Returns:
        boot_data (array): Set of sampled empirical relation coefficients.
    '''
    if boot_spec=='default':
        boot_data = np.genfromtxt('./priors/sampled_empirical_relation_coefficients.tab',skip_header=1)
    else:
        boot_data = np.genfromtxt(boot_spec,skip_header=1)
    return boot_data

def gen_simulated_eventdict(Mlist,Rtrue,sigma_f,nsignal,nnoise,rng,scatter=None,use_prior='array',priordir=None,prior=None,
                            kde_boundary='Standard',kde_bandwidth=0.15,nosignal=False,plot='none',glow=True,saveto=None,showplot=True):
    '''
    Function to create an dictionary of **simulated** events with associated parameters and data. Allows for user-specified
    signal and noise. Simulated peak frequency recoveries consist of a user-specified combination of gaussian signal with width
    sigma_f and mean = empirical_relation_f(Rtrue,M) for each M given in Mlist. 
    
    See gen_BayesWave_eventdict() for the real/realistic case using BayesWave data
    
    Arguments:
        Mlist (list of floats) : List of chirp masses to use for the simulation, one for each event.
        Rtrue (float) : "True" value of R_1.6 to be injected. 
        sigma_f (float) : Standard deviation of simulated peak frequency signal samples.
        nsignal (int) : Number of signal samples to simulate.
        nnoise (int) : Number of noise samples to add to the signal.
        rng (Generator) : Numpy RNG instance (e.g. np.random.default_rng(seed))
        scatter (float) : Desired scatter in kHz of fpeak values from empirical relation predictions.
                          Recommended: 0.053 kHz (std. dev. of Vretinaris relation residuals)
        use_prior (str) : Specifies type of peak frequency prior to use. Can be 'array' (uses pre-loaded prior samples; 
                          you will then need to specify prior=), 'load' (dynamically loads a peak frequency prior; you 
                          will then need to specify priordir=), or 'uniform' (sets a uniform peak frequency prior).
        priordir (str) : See use_prior. Path to directory with peak frequency prior samples.
        prior (array) : See use_prior. Array of peak frequency prior samples.
        kde_boundary (str) : KDE boundary conditions. Can be 'Standard' or 'Reflection'. See Posterior_f() for details.
        kde_bandwidth (float) : KDE bandwidth passed to Posterior_f(). See Posterior_f() for details.
        nosignal (bool) : If True, produces an eventdict identical to the usual one produced, but all peak frequency samples
                          are instead drawn at random from the specified prior. Useful for characterizing how the analysis
                          behaves when considering data consisting only of noise.
        plot (str) : Whether to plot some events. Can be 'none', 'smallset' (first 4), or 'subset' (first 10).
        glow (bool) : If plotting, whether to use mplcyberpunk glow and shading effects.
        saveto (str) : If not None, should be set to '/path/to/save/figure.pdf/'
        
    Return:
        eventdict (dict) : Dictionary with simulated peak frequency recoveries and other relevant information stored for analysis.
    '''
    ## initialize event dict
    eventdict = {}
    ## make sub dict for sim parameters
    sim_param_dict = {'nnoise':nnoise,'nsignal':nsignal,'sigma_f':sigma_f,'rng':rng}
    ## deal with duplicate+ M values
    Mstrlist = ['{:0.2f}'.format(M) for M in Mlist]
    for Mi in Mstrlist:
        dupes = [jj for jj, Mj in enumerate(Mstrlist) if Mi==Mj]
        if len(dupes) > 1:
            for count, kdx in enumerate(dupes):
                Mstrlist[kdx] = Mstrlist[kdx]+'-'+str(count+1)
    for M, mstr in zip(Mlist,Mstrlist):
        ## initalize event entry in eventdict
        eventdict[mstr] = {}
        eventdict[mstr]['mchirp'] = M
        ## get true fpeak
        ftrue_i = empirical_relation_f(Rtrue,M) # in kHz
        if scatter is not None:
            ftrue_i = ftrue_i + st.norm.rvs(loc=0,scale=scatter,random_state=rng)
        eventdict[mstr]['ftrue'] = ftrue_i
        
        ## simulate fpeak recovery
        if nosignal==False:
            fpeaksignal_i = st.norm.rvs(loc=ftrue_i,scale=sigma_f,size=nsignal,random_state=rng)
            fpeaksignal_i = fpeaksignal_i[(fpeaksignal_i>1.5)&(fpeaksignal_i<4)]
            ndiff = nsignal - len(fpeaksignal_i)
            if use_prior=='load':
                eventdict[mstr]['event_type'] = 'simulated run with signals (BayesWave prior)'
                fpeaknoise_i = load_BayesWave_fprior(priordir,'H1L1',rng,draws=nnoise+ndiff,fmin=1500)/1000
            elif use_prior=='array':
                eventdict[mstr]['event_type'] = 'simulated run with signals (BayesWave prior)'
                fpeaknoise_i = draw_from_BayesWave_fprior(prior,rng,draws=nnoise+ndiff)/1000
            elif use_prior=='uniform':
                eventdict[mstr]['event_type'] = 'simulated run with signals (uniform prior)'
                fpeaknoise_i = st.uniform.rvs(loc=1.5,scale=2.5,size=nnoise+ndiff,random_state=rng)
            else:
                raise TypeError("Invalid prior specification (need something to draw from for simulated signals!)")
            fpeakchains_i = np.concatenate((fpeaksignal_i,fpeaknoise_i.flatten()))
        elif nosignal==True:
            if use_prior=='load':
                eventdict[mstr]['event_type'] = 'simulated run without signals (BayesWave prior)'
                fpeaknoise_i = load_BayesWave_fprior(priordir,'H1L1',rng,draws=nnoise,fmin=1500)/1000
            elif use_prior=='array':
                eventdict[mstr]['event_type'] = 'simulated run without signals (BayesWave prior)'
                fpeaknoise_i = draw_from_BayesWave_fprior(prior,rng,draws=nnoise)/1000
            elif use_prior=='uniform':
                eventdict[mstr]['event_type'] = 'simulated run without signals (uniform prior)'
                fpeaknoise_i = st.uniform.rvs(loc=1.5,scale=2.5,size=nnoise,random_state=rng)
            else:
                raise TypeError("Invalid prior specification (need something to draw from for simulated signals!)")
            fpeakchains_i = fpeaknoise_i
        else:
            raise TypeError("Invalid specification. nosignal must be True or False.")

        eventdict[mstr]['fchains'] = fpeakchains_i
        ## KDE
        ## new version using Posterior_f class
        eventdict[mstr]['kde'] = Posterior_f(fpeakchains_i,kde_boundary,kde_bandwidth=kde_bandwidth)
        ## save simulation parameters
        eventdict[mstr]['params'] = sim_param_dict

    if plot=='subset':
        fs = np.linspace(1.5,4,100)
        fig, axes = plt.subplots(5,2,figsize=(10,15),sharex=True,sharey=False)
        for (i,ax), M, mstr in zip(enumerate(axes.flatten()),Mlist[:10],Mstrlist[:10]):
            ax.plot(fs,eventdict[mstr]['kde'].pdf(fs),label='KDE')
            ax.hist(eventdict[mstr]['fchains'],density=True,alpha=0.7,bins=50,label='Samples')
            ax.axvline(eventdict[mstr]['ftrue'],label='True $\mathrm{f_{peak}}$',color='k',ls='--')
            ax.set_xlabel('$\mathrm{f_{peak}}$ (Hz)')
            ax.set_xlim(1.5,4)
            ax.set_yticks([])
            ax.set_title('$\mathrm{f_{peak}}$ Posterior for $\mathcal{M}$='+mstr.split('-')[0]+' $M_{\odot}$',fontsize=12)
            if i==1:
                ax.legend()
            if glow==True:
                mplcyberpunk.add_glow_effects(ax)
        plt.tight_layout(rect=(0,0,1,.95))
        plt.suptitle('$\mathrm{f_{peak}}$ Posteriors for 10 Simulated Events',fontsize=16)
        if saveto is not None:
            plt.savefig(saveto,bbox_inches='tight')
        if showplot==True:
            plt.show()
        else:
            plt.close()
    elif plot=='smallset':
        fs = np.linspace(1.5,4,100)
        fig, axes = plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=False)
        for (i,ax), M, mstr in zip(enumerate(axes.flatten()),Mlist[:4],Mstrlist[:4]):
            ax.plot(fs,eventdict[mstr]['kde'].pdf(fs),label='KDE')
            ax.hist(eventdict[mstr]['fchains'],density=True,alpha=0.7,bins=50,label='Samples')
            ax.axvline(eventdict[mstr]['ftrue'],label='True $\mathrm{f_{peak}}$',color='k',ls='--')
            ax.set_xlabel('$\mathrm{f_{peak}}$ (Hz)')
            ax.set_xlim(1.5,4)
            ax.set_yticks([])
            ax.set_title('$\mathrm{f_{peak}}$ Posterior for $\mathcal{M}$='+mstr.split('-')[0]+' $M_{\odot}$',fontsize=12)
            if i==1:
                ax.legend()
            if glow==True:
                mplcyberpunk.add_glow_effects(ax)
        plt.tight_layout(rect=(0,0,1,.95))
        plt.suptitle('$\mathrm{f_{peak}}$ Posteriors for 4 Simulated Events',fontsize=16)
        if saveto is not None:
            plt.savefig(saveto,bbox_inches='tight')
        if showplot==True:
            plt.show()
        else:
            plt.close()
    elif plot=='smallset_compact':
        fs = np.linspace(1.5,4,100)
        fig, axes = plt.subplots(2,2,figsize=(6,6),sharex=True,sharey=False)
        for (i,ax), M, mstr in zip(enumerate(axes.flatten()),Mlist[:4],Mstrlist[:4]):
            ax.plot(fs,eventdict[mstr]['kde'].pdf(fs),label='KDE')
            ax.hist(eventdict[mstr]['fchains'],density=True,alpha=0.7,bins=50,label='Samples')
            ax.axvline(eventdict[mstr]['ftrue'],label='True $\mathrm{f_{peak}}$',color='k',ls='--')
            
#             ax.set_xlabel('$\mathrm{f_{peak} (kHz)}$')
            ax.set_xlim(1.5,4)
            ax.set_xticks(ticks=[1.5,2,2.5,3,3.5,4],labels=['','2.0','','3.0','','4.0'])
            ax.set_yticks([])
#             ax.set_title('$\mathrm{f_{peak}}$ Posterior for $\mathcal{M}$='+mstr.split('-')[0]+' $M_{\odot}$',fontsize=12)
            ax.set_title('$\mathcal{M}$='+mstr.split('-')[0]+' $M_{\odot}$',fontsize=12)
            if i==1:
                ax.legend()
            if glow==True:
                mplcyberpunk.add_glow_effects(ax)
        axes.flatten()[2].set_xlabel('$\mathrm{f_{peak} (kHz)}$')
        axes.flatten()[3].set_xlabel('$\mathrm{f_{peak} (kHz)}$')
        plt.tight_layout(rect=(0,0,1,.95))
        plt.suptitle('$\mathrm{f_{peak}}$ Posteriors for 4 Simulated Events',fontsize=16)
        if saveto is not None:
            plt.savefig(saveto,bbox_inches='tight')
        if showplot==True:
            plt.show()
        else:
            plt.close()
    return eventdict


def gen_BayesWave_eventdict(events,ev_df,seed,event_masses='simulated',Mpost_type='samples',Mchirp_scaling='none',
                            kde_boundary='Standard',kde_bandwidth=0.15,use_prior='array',
                            priordir=None,prior=None,obs_run=False,sim_df=None,z_adj=None,nosignal=False,
                            aggregation='sum',ifos=['H1','L1'],
                            plot='none',title=None,glow=False,saveto=None,showplot=True):
    '''
    Function to create an dictionary of BayesWave-analyzed events with associated parameters and data. 
    Allows for user specification of peak frequency priors and includes functionality to interface with the .csv files
    produced by the LIGO O4 and O5 observing run simulations produced by Petrov et al. (2021) https://arxiv.org/abs/2108.07277
    Allows for use of Theo Soultanis' post-merger NR simulations. Option to "dry run" analysis over signal-less data.
    
    Arguments:
        events (list of strs) : List with paths to the containing folders for each event's BayesWave output data. 
                                (Use of glob.glob is recommended)
        ev_df (pandas DataFrame) : DataFrame containing information about the Soultanis NR simulations.
        event_masses (str or list of strs) : Can be 'simulated' or a list with paths to the containing folders for each event's
                                             Bilby chirp mass posterior chains. 
                                             IMPORTANT NOTE: in the path list case, these paths MUST correspond to and be in the same order
                                             as the fpeak events specified in the first argument! Best practice is to just place
                                             a folder with chirp mass posterior samples for each event in the same folder as your 
                                             fpeak posterior samples for that event, with a consistent naming scheme.
        Mpost_type (str) : Can be 'samples' or 'distribution'.
                           If 'samples', posterior samples will be generated from a normal distribution and fit with a KDE.
                           If 'distribution', M posteriors will just be normal distributions (scipy.stats.norm).
        Mchirp_scaling (str) : How to scale the simulated chirp mass posterior width. Can be 'none' (sets sigma_Mc = 0.01 Msun),
                               'dist' (scales with distance to merger), or 'snr' (scales with network signal-to-noise)
                               We use GW170817 as a reference point. (See Farr et al. (2016) for details on this scaling)
                               Only needed if event_masses=='simulated'.
        seed : Seed for Numpy RNG instance (e.g. np.random.default_rng(seed))
        kde_boundary (str) : KDE boundary conditions. Can be 'Standard' or 'Reflection'. See Posterior_f() for details.
        kde_bandwidth (float) : KDE bandwidth passed to Posterior_f(). See Posterior_f() for details.
        use_prior (str) : Specifies type of peak frequency prior to use. Can be 'array' (uses pre-loaded prior samples; 
                          you will then need to specify prior=), 'load' (dynamically loads a peak frequency prior; you 
                          will then need to specify priordir=), or 'uniform' (sets a uniform peak frequency prior).
        priordir (str) : See use_prior. Path to directory with peak frequency prior samples.
        prior (array) : See use_prior. Array of peak frequency prior samples or dict with one array per ifo, with the ifos as keys.
        obs_run (bool) : If True, allows user to load in specific event parameters from a Petrov et al. (2021) simulation.
                         You will then need to specify sim_df.
        sim_df (pandas DataFrame) : DataFrame containing information about the Petrov et al. (2021) simulations.
        z_adj (str) : Whether to account for redshift. Only applicable if obs_run is True.
                      If 'known', uses luminosity distances from ev_df_file 
                      to compute redshift correction for each event, assuming standard LambdaCDM. 
                      ('posteriors' coming soon (TM), which will allow for use of inspiral redshift posteriors.)
        nosignal (bool) : If True, produces an eventdict identical to the usual one produced, but all peak frequency samples
                          are instead drawn at random from the specified prior. Useful for characterizing how the analysis
                          behaves when considering data consisting only of noise.
        aggregation (str) : How to combine fpeak posteriors from different detectors. Can be 'sum' (adds samples together) or 'mult' (multiplies posterior kdes).
        ifos (list of str) : Which interferometers to use data from. Should be a list of strings. Can be any combination of H1, L1, and/or V1, e.g. ['H1','L1','V1'].
        plot (str) : Whether to plot some events. Can be 'none', 'smallset' (first 4), or 'subset' (first 10).
        title (str) : If plotting, what to set as the plot title.
        glow (bool) : If plotting, whether to use mplcyberpunk glow and shading effects.
        saveto (str) : If you wish to save the subset/smallset plot, should be set to '/path/to/save/figure.pdf/'
        
    Return:
        eventdict (dict) : Dictionary with peak frequency recoveries and other relevant information stored for analysis.
    '''
    rng = np.random.default_rng(seed)
    ## initialize event dict
    eventdict = {}
    ## load fpeak posteriors and make kdes. Store in dict and plot
    if event_masses=='simulated':
        ## GW170817 parameters (for scaling)
        M170817 = 1.188
        sigma170817 = 0.004
        D170817 = 40
        SNR170817 = 32.4
        for event in tqdm(events,total=len(events),ascii='>='):
            ## initalize event entry in eventdict
            mstr = event.split('/')[-1].split('_')[1]
            ## get chirp mass
            m1, m2 = parse_mstr(mstr)
            M = calc_Mchirp(m1,m2)
            ## add distance if doing observing run sims
            ## also deal with duplicate events at different distances
            if obs_run==True:
                dist_str = event.split('/')[-1].split('_')[-1].replace('Mpc','')
#                dist_i = float(dist_str.replace('-2',''))
                dist_i = float(dist_str.split('-')[0])
                key = mstr+'_'+dist_str
                eventdict[key] = {}
                eventdict[key]['mchirp'] = M
                eventdict[key]['dist'] = dist_i
                if Mchirp_scaling=='dist':
                    Msigma = (M/M170817)*(dist_i/D170817)*sigma170817
                elif Mchirp_scaling=='snr':
                    ## get SNR
                    if sim_df is not None:
                        SNR_i = sim_df[np.abs(sim_df['distance'] - eventdict[key]['dist'] )< 0.01]['snr'].to_numpy()
                        eventdict[key]['snr'] = SNR_i[0]
                    M = eventdict[key]['mchirp']
                    SNR = eventdict[key]['snr']
                    Msigma = (M/M170817)*(SNR170817/SNR)*sigma170817
                else:
                    Msigma = 0.01
                Mkern_base = st.norm(loc=M,scale=Msigma)
                if Mpost_type=='distribution':
                    eventdict[key]['Mkern'] = Mkern_base
                elif Mpost_type=='samples':
                    Msamples = Mkern_base.rvs(size=1000)
                    eventdict[key]['Mkern'] = Posterior_M(Msamples,'Standard',bw_method=None)
                    
                
                ## get true fpeak
                ftrue_i = ev_df[parse_datafile(ev_df['file']) 
                                == event.split('/')[-1].split('_')[1]]['fpeak'].to_numpy()/1000 # in kHz
                eventdict[key]['ftrue'] = ftrue_i
                ## get redshift
                if z_adj is not None:
                    if z_adj == 'known':
                        cosmo = co.FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.3)
                        z = co.z_at_value(cosmo.luminosity_distance,dist_i*u.Mpc)
                        eventdict[key]['z'] = z
                        eventdict[key]['ftrue_redshifted'] = ftrue_i/(1+z)
                    else:
                        raise TypeError("Invalid specification of z_adj. Only 'known' is currently supported.")
                
                    
                ## load fpeak recovery
                if nosignal==False:
                    eventdict[key]['event_type'] = 'obs run with signals'
                    fpeakchains_i = load_BayesWave_fpeak(event,rng,ifos=ifos,nsamp=50000,bwfmin=1000,bwfmax=5000,
                                                         use_prior=use_prior,prior=prior,priordir=priordir)/1000
                elif nosignal==True:
                    for i, ifo in enumerate(ifos):
                        if use_prior=='load':
                            eventdict[key]['event_type'] = 'obs run without signals (BayesWave prior)'
                            fsamp_i = load_BayesWave_fprior(priordir,'H1L1',rng,draws=5000,fmin=1500)/1000
                        elif use_prior=='array':
                            if type(prior) is dict:
                                prior_i = prior[ifo]
                            else:
                                prior_i = prior
                            eventdict[key]['event_type'] = 'obs run without signals (BayesWave prior)'
                            fsamp_i = draw_from_BayesWave_fprior(prior_i,rng,draws=5000)/1000
                        elif use_prior=='uniform':
                            eventdict[key]['event_type'] = 'obs run without signals (uniform prior)'
                            fsamp_i = st.uniform.rvs(loc=1.5,scale=2.5,size=5000,random_state=rng)
                        else:
                            print("Invalid prior specification for nosignal=True (need something to draw from!)")
                        ## one column for each detector
                        if i==0:
                            fpeakchains_i = fsamp_i.reshape(-1,1)
                        else:
                            fpeakchains_i = np.hstack((fpeakchains_i, fsamp_i.reshape(-1,1)))
                else:
                    raise TypeError("Invalid specification. nosignal must be True or False.")
                
                if aggregation == 'sum':
                    fpeakchains_i = fpeakchains_i.flatten()
                
                eventdict[key]['fchains'] = fpeakchains_i
                ## KDE
                ## new version using Posterior_f class
                eventdict[key]['kde'] = Posterior_f(fpeakchains_i,kde_boundary,kde_bandwidth=kde_bandwidth)

            else:
                eventdict[mstr] = {}
                eventdict[mstr]['mchirp'] = M
                ## get true fpeak
                ftrue_i = ev_df[parse_datafile(ev_df['file']) 
                                == event.split('/')[-1].split('_')[1]]['fpeak'].to_numpy()/1000 # in kHz
                eventdict[mstr]['ftrue'] = ftrue_i
                ## get true fpeak
                ftrue_i = ev_df[parse_datafile(ev_df['file']) 
                                == event.split('/')[-1].split('_')[1]]['fpeak'].to_numpy()/1000 # in kHz
                eventdict[mstr]['ftrue'] = ftrue_i
                ## load fpeak recovery
                if nosignal==False:
                    eventdict[mstr]['event_type'] = 'non-obs-run with signals'
                    fpeakchains_i = load_BayesWave_fpeak(event,rng,ifos=ifos,nsamp=2500,fmin=1500,
                                                         use_prior=use_prior,prior=prior,priordir=priordir)/1000
                elif nosignal==True:
                    if use_prior=='load':
                        eventdict[mstr]['event_type'] = 'non-obs-run without signals (BayesWave prior)'
                        fpeakchains_i = load_BayesWave_fprior(priordir,'H1L1',rng,draws=5000,fmin=1500)/1000
                    elif use_prior=='array':
                        eventdict[mstr]['event_type'] = 'non-obs-run without signals (BayesWave prior)'
                        fpeakchains_i = draw_from_BayesWave_fprior(prior,rng,draws=5000)/1000
                    else:
                        raise TypeError("Invalid prior specification for nosignal=True (need something to draw from!)")
                else:
                    raise TypeError("Invalid specification. nosignal must be True or False.")
                
                if aggregation == 'sum':
                    fpeakchains_i = fpeakchains_i.flatten()
                
                eventdict[mstr]['fchains'] = fpeakchains_i
                ## KDE
                ## new version using Posterior_f class
                eventdict[mstr]['kde'] = Posterior_f(fpeakchains_i,kde_boundary,kde_bandwidth=kde_bandwidth)
    else:
        if len(events)!=len(event_masses):
            raise TypeError("Different numbers of specified fpeak events and chirp mass posteriors!")
        for event, mpost in tqdm(zip(events,event_masses),total=len(events),ascii='>='):
            ## initalize event entry in eventdict
            mstr = event.split('/')[-1].split('_')[1]
            ## get chirp mass
            m1, m2 = parse_mstr(mstr)
            M = calc_Mchirp(m1,m2)
            ## add distance if doing observing run sims
            ## also deal with duplicate events at different distances
            if obs_run==True:
                dist_str = event.split('/')[-1].split('_')[-1].replace('Mpc','')
                dist_i = float(dist_str.replace('-2',''))
                key = mstr+'_'+dist_str
                eventdict[key] = {}
                eventdict[key]['mchirp'] = M
                eventdict[key]['dist'] = dist_i
                ## get true fpeak
                ftrue_i = ev_df[parse_datafile(ev_df['file']) 
                                == event.split('/')[-1].split('_')[1]]['fpeak'].to_numpy()/1000 # in kHz
                eventdict[key]['ftrue'] = ftrue_i
                ## get SNR
                if sim_df is not None:
                    SNR_i = sim_df[np.abs(sim_df['distance'] - eventdict[key]['dist'] )< 0.01]['snr'].to_numpy()
                    eventdict[key]['snr'] = SNR_i[0]
                ## load fpeak recovery
                if nosignal==False:
                    eventdict[key]['event_type'] = 'obs run with signals'
                    fpeakchains_i = load_BayesWave_fpeak(event,rng,ifos=ifos,nsamp=2500,fmin=1500,
                                                         use_prior=use_prior,prior=prior,priordir=priordir)/1000
                elif nosignal==True:
                    if use_prior=='load':
                        eventdict[key]['event_type'] = 'obs run without signals (BayesWave prior)'
                        fpeakchains_i = load_BayesWave_fprior(priordir,'H1L1',rng,draws=5000,fmin=1500)/1000
                    elif use_prior=='array':
                        eventdict[key]['event_type'] = 'obs run without signals (BayesWave prior)'
                        fpeakchains_i = draw_from_BayesWave_fprior(prior,rng,draws=5000)/1000
                    elif use_prior=='uniform':
                        eventdict[key]['event_type'] = 'obs run without signals (uniform prior)'
                        fpeakchains_i = st.uniform.rvs(loc=1.5,scale=2.5,size=5000,random_state=rng)
                    else:
                        print("Invalid prior specification for nosignal=True (need something to draw from!)")
                else:
                    raise TypeError ("Invalid specification. nosignal must be True or False.")
                
                if aggregation == 'sum':
                    fpeakchains_i = fpeakchains_i.flatten()
                
                eventdict[key]['fchains'] = fpeakchains_i
                ## load chirp mass recovery and store
                mchirpchains_i = load_Bilby_Mchirp(mpost)
                eventdict[key]['mchains'] = mchirpchains_i
                ## KDE
                ## initialize using Posterior_f class
                eventdict[key]['kde'] = Posterior_f(fpeakchains_i,kde_boundary,kde_bandwidth=kde_bandwidth)
                ## do the same for Mchirp
                eventdict[key]['mkern'] = Posterior_M(mchirpchains_i,'Standard')

            else:
                eventdict[mstr] = {}
                eventdict[mstr]['mchirp'] = M
                ## get true fpeak
                ftrue_i = ev_df[parse_datafile(ev_df['file']) 
                                == event.split('/')[-1].split('_')[1]]['fpeak'].to_numpy()/1000 # in kHz
                eventdict[mstr]['ftrue'] = ftrue_i
                ## get true fpeak
                ftrue_i = ev_df[parse_datafile(ev_df['file']) 
                                == event.split('/')[-1].split('_')[1]]['fpeak'].to_numpy()/1000 # in kHz
                eventdict[mstr]['ftrue'] = ftrue_i
                ## load fpeak recovery
                if nosignal==False:
                    eventdict[mstr]['event_type'] = 'non-obs-run with signals'
                    fpeakchains_i = load_BayesWave_fpeak(event,rng,ifos=ifos,nsamp=2500,fmin=1500,
                                                         use_prior=use_prior,prior=prior,priordir=priordir)/1000
                elif nosignal==True:
                    if use_prior=='load':
                        eventdict[mstr]['event_type'] = 'non-obs-run without signals (BayesWave prior)'
                        fpeakchains_i = load_BayesWave_fprior(priordir,'H1L1',rng,draws=5000,fmin=1500)/1000
                    elif use_prior=='array':
                        eventdict[mstr]['event_type'] = 'non-obs-run without signals (BayesWave prior)'
                        fpeakchains_i = draw_from_BayesWave_fprior(prior,rng,draws=5000)/1000
                    else:
                        raise TypeError("Invalid prior specification for nosignal=True (need something to draw from!)")
                else:
                    raise TypeError("Invalid specification. nosignal must be True or False.")
                
                if aggregation == 'sum':
                    fpeakchains_i = fpeakchains_i.flatten()
                
                eventdict[mstr]['fchains'] = fpeakchains_i
                ## KDE
                ## new version using Posterior_f class
                eventdict[mstr]['kde'] = Posterior_f(fpeakchains_i,kde_boundary,kde_bandwidth=kde_bandwidth)
                
                ## load chirp mass recovery and store
                mchirpchains_i = load_Bilby_Mchirp(mpost)
                eventdict[mstr]['mchains'] = mchirpchains_i
                eventdict[mstr]['mkern'] = Posterior_M(mchirpchains_i,'Standard')
    if plot=='subset':
        fs = np.linspace(1.5,4,100)
        fig, axes = plt.subplots(5,2,figsize=(10,15),sharex=True,sharey=False)
        for (i,ax), key in zip(enumerate(axes.flatten()),[*eventdict.keys()][:10]):
#             mstr = event.split('/')[-1].split('_')[1]
            ax.plot(fs,eventdict[key]['kde'].pdf(fs),label='KDE')
            ax.hist(eventdict[key]['fchains'].flatten(),density=True,alpha=0.7,bins=50,label='Samples')
            if z_adj is not None:
                ax.set_title('$\mathrm{f_{peak}}$ Posterior for $\mathcal{M}$='+"{:0.2f}".format(eventdict[key]['mchirp'])
                         +' $M_{\odot}$'+' (z={:0.2f})'.format(eventdict[key]['z']),fontsize=12)
                ax.axvline(eventdict[key]['ftrue_redshifted'],label='True $\mathrm{f_{peak}}$',color='k',ls='--')
            else:
                ax.set_title('$\mathrm{f_{peak}}$ Posterior for $\mathcal{M}$='+"{:0.2f}".format(eventdict[key]['mchirp'])
                         +' $M_{\odot}$',fontsize=12)
                ax.axvline(eventdict[key]['ftrue'],label='True $\mathrm{f_{peak}}$',color='k',ls='--')
            ax.set_xlabel('$\mathrm{f_{peak}}$ (Hz)')
            ax.set_xlim(1.5,4)
            ax.set_yticks([])
            if i==1:
                ax.legend()
            if glow==True:
                mplcyberpunk.add_glow_effects(ax)
        plt.tight_layout(rect=(0,0,1,.95))
        if title is not None:
            plt.suptitle(title)
        else:
            plt.suptitle('$\mathrm{f_{peak}}$ Recoveries for 10 BayesWave Runs',fontsize=16)
        if saveto is not None:
            plt.savefig(saveto,bbox_inches='tight')
        if showplot==True:
            plt.show()
        else:
            plt.close()
    return eventdict    
    


## SECTION 5: LIKELIHOODS

## Contains functions to perform likelihood calculations

def selection_function(fhat):
    '''
    Computes the selection function coefficients a and b. 
    The selection function ensures no probability is assigned to unobservable predicted values of fpeak.
    For further information, see derivation in Criswell et al. (2022)
    
    Arguments:
        fhat (array) : Empirical relation predicted values of the peak frequency
    
    Returns:
        a,b (float) : Selection function coefficients
    '''
    a = (fhat>1.5) & (fhat<4)
    b = np.invert(a)
    return a,b
    
def single_event_likelihood(Rs,Ms,Mkern,fkern,fprior,Mprior,z=None,bootstrap=None):
    '''
    Function to compute the likelihood L(data|R16) for a single event.
    Expanded into L(d_PM,d_IN|R16) = p(fp|d_PM)/pi(fp) * p(Mc|d_IN)/pi(Mc) * p(fp,Mc|R16)
    
    Arguments:
        Rs (array) : R16 values for which to find L(data|R)
        Ms (array) : Chirp mass values for which to find fhat = empirical_relation_f(R,Ms)
        Mkern (kernel) : Chirp mass posterior
        fkern (kernel) : Peak frequency posterior
        fprior (kernel) : Peak frequency prior
        Mprior (kernel) : Chirp mass prior
        z (float) : If None, no redshift adjustment is made. If specified, shoudl be the (known) redshift of BNS merger.
        bootstrap (array) : If None, empirical relation is assumed to be exact. If specified, must be an array of boostrapped 
                            empirical relation coefficients. See empirical_relation_bootstrap() for details.
    
    Returns:
        Rlikes (array) : likelihood p(data|R) for each R given in Rs
    '''
    ## initialize likelihood array
    Rlikes = []
    ## get number of detectors
    Ndet = fkern.ndet
    if z is not None:
        z_adj = 1/(1+z)
    else:
        z_adj = 1
    for R in Rs:
        ## fhat = F_E(R,M)
        if bootstrap is not None:
            Ms_broad = np.repeat(Ms.reshape(-1,1),bootstrap.shape[0],axis=1)
            fhat_MR = empirical_relation_bootstrap(R,Ms_broad,bootstrap)*z_adj
        else:
            fhat_MR = empirical_relation_f(R,Ms)*z_adj
        ## p(f,M|R) = delta(fhat - f)*S(f,M|R)
        a,b = selection_function(fhat_MR)

        ## p(D|M,f) if fhat is detectable
        ## = p(f|D_PM)p(D_PM)/p(f) * p(M|D_IN)p(D_IN)/p(M)
        ## but factor p(D_x) out
        if bootstrap is not None:
            undet_like = np.repeat((Mkern.pdf(Ms)/Mprior.pdf(Ms) \
                                    *1).reshape(-1,1),bootstrap.shape[0],axis=1)[b]
                                    #*st.norm.pdf(Ms,loc=1.155,scale=0.05)).reshape(-1,1),bootstrap.shape[0],axis=1)[b]
        else:
            undet_like = (Mkern.pdf(Ms[b])/Mprior.pdf(Ms[b]))
        ## make sure that these give same result under no information in f
        if np.sum(a) > 0:
            if bootstrap is not None:
                Mdet_frac = np.repeat((Mkern.pdf(Ms)/Mprior.pdf(Ms) \
                                       *1).reshape(-1,1),bootstrap.shape[0],axis=1)[a]
                                      # *st.norm.pdf(Ms,loc=1.155,scale=0.05)).reshape(-1,1),bootstrap.shape[0],axis=1)[a]
                det_like = (fkern.pdf(fhat_MR[a])/(fprior.pdf(fhat_MR[a])))*Mdet_frac
            else:
                det_like = (fkern.pdf(fhat_MR[a])/(fprior.pdf(fhat_MR[a])))*(Mkern.pdf(Ms[a])/Mprior.pdf(Ms[a]))
            ## marginalize
            Rlike_i = np.sum(det_like) + np.sum(undet_like)
#            norm = 1#np.sum(fkern.pdf(fhat_MR[a])/fprior.pdf(fhat_MR[a]))/np.sum(a)
            ## this handles the rare case where all observable values of fhat have p(fhat)=0
            ## (only really crops up when fpeak is extremely well-localized)
#            if norm==0:
#                Rlike_i = np.sum(det_like) + np.sum(undet_like)
#            else:
#                Rlike_i = np.sum(det_like)/norm + np.sum(undet_like)

        else:
            Rlike_i = np.sum(undet_like)

        Rlikes.append(Rlike_i)
    
    return np.array(Rlikes)

def get_aggregate_likelihood(likelihood_list):
    '''
    Function to take product of a list of likelihoods to compute a multi-event likelihood. 
    Contains some safety checks as well.
    
    Arguments:
        likelihood_list (list of arrays) : list of N individual event likelihoods
    
    Returns:
        aggregate_likelihood (array) : Multiple event likelihood p(data_{i...N}|R16)
    '''
    if len(likelihood_list)==1:
        return likelihood_list[0]
    elif len(likelihood_list)==0:
        raise TypeError('Empty list!')
    else:
        unnormed_aggregate = np.prod(likelihood_list/np.sum(likelihood_list,axis=1).reshape(-1,1),axis=0)
        ## even checking for nans and infs does not fully catch the transition into numerical errors
        ## add artificial cutoff to handle this
        if np.any(np.isnan(unnormed_aggregate)) or np.any(np.isinf(unnormed_aggregate)) or np.any(unnormed_aggregate<1e-300):
            aggregate = iterative_normalized_aggregate_likelihood(likelihood_list)
        else:
            aggregate = unnormed_aggregate/np.sum(unnormed_aggregate)
        return aggregate
    


def iterative_normalized_aggregate_likelihood(likelihood_list):
    '''
    Computes the posterior product of a likelihood list, normalizing as it goes.
    This method is slower than other approaches, but in the case of large N it 
    ensures that the outcome will not be erroneously zero or NaN due to overflow.
    
    Arguments:
        likelihood_list (list of arrays) : list of N individual event likelihoods
    
    Returns:
        aggregate_likelihood (array) : Multiple event likelihood p(data_{i...N}|R16)
    '''
    if len(likelihood_list)==1:
        return likelihood_list[0]
    elif len(likelihood_list)==0:
        raise TypeError('Empty list!')
    
    for i, likelihood in enumerate(likelihood_list):
        if i==0:
            aggregate = likelihood/np.sum(likelihood)
        else:
            normed_likelihood = likelihood/np.sum(likelihood)
            unnormed_aggregate = aggregate*normed_likelihood
            aggregate = unnormed_aggregate/np.sum(unnormed_aggregate)
    
    return aggregate


def get_multievent_likelihoods(Rs,Ms,eventdict,Mchirp_type='simulated',fprior=st.uniform(loc=1.5,scale=2.5),
                               Mprior=st.uniform(loc=0,scale=5),Mchirp_scaling='none',verbose=True,bootstrap=None,z_adj=None):
    '''
    Function to compute a list containing the likelihood p(data|R) for each event in an eventdict produced by either
    gen_BayesWave_eventdict() or gen_simulated_eventdict(). 
    Note: currently set up to simulate chirp mass distributions using realistic scaling (see Farr et al. 2016). 
        - will need to be updated to use samples instead
    
    Arguments:
        Rs, Ms (array) : R and M ranges as used throughout the code.
        eventdict (dict) : Dictionary containing data for and information about each event to be analyzed. 
                           See gen_BayesWave_eventdict() and/or gen_simulated_eventdict() for details.
        Mchirp_type (str) : Nature of chirp mass posterior. Can be 'simulated' or 'samples'. If 'simulated', the Mchirp value stored in
                            each eventdict entry will be used (along with Mchirp_scaling, below) to simulate the Mchirp posteriors as 
                            Normal distributions. If 'samples', the Posterior_M object stored in each eventdict will be used instead.
        fprior (kernel) : Peak frequency prior kernel.
        Mprior (kernel) : Chirp mass prior kernel.
        Mchirp_scaling (str) : How to scale the simulated chirp mass posterior width. Can be 'none' (sets sigma_Mc = 0.01 Msun),
                               'dist' (scales with distance to merger), or 'snr' (scales with network signal-to-noise)
                               We use GW170817 as a reference point. (See Farr et al. (2016) for details on this scaling)
                               Only needed if Mchirp_type is 'simulated'.
        verbose (bool) : If True, code will print progress updates with each event.
        bootstrap (array) : If None, empirical relation is assumed to be exact. If specified, must be an array of boostrapped 
                            empirical relation coefficients. See empirical_relation_bootstrap() for details.
        z_adj (str) : Whether to account for redshift. If 'known', uses luminosity distances from ev_df_file 
                      to compute redshift correction for each event, assuming standard LambdaCDM. 
                      ('posteriors' coming soon (TM), which will allow for use of inspiral redshift posteriors.)
    
    Returns:
        likes_all (list of arrays) : List containing the likelihood p(data_i|R16) for each event considered, in eventdict order.
    '''
    likes_all = []
    ## GW170817 parameters (for scaling)
    M170817 = 1.188
    sigma170817 = 0.004
    D170817 = 40
    SNR170817 = 32.4
    for i,key in tqdm(enumerate(eventdict.keys()),total=len(eventdict),ascii='>='):
        if verbose==True:
            print('Processing event {}/{} (Mc={:0.2f} Mo)'.format(i+1,len(eventdict.keys()),eventdict[key]['mchirp']))
        if Mchirp_type=='simulated':
            if Mchirp_scaling=='dist':
                M = eventdict[key]['mchirp']
                D = eventdict[key]['dist']
                Msigma = (M/M170817)*(D/D170817)*sigma170817
            elif Mchirp_scaling=='snr':
                M = eventdict[key]['mchirp']
                SNR = eventdict[key]['snr']
                Msigma = (M/M170817)*(SNR170817/SNR)*sigma170817
            else:
                Msigma = 0.01
            Mkern = st.norm(loc=eventdict[key]['mchirp'],scale=Msigma)
        elif Mchirp_type=='samples':
            Mkern = eventdict[key]['Mkern']
        else:
            raise TypeError("Invalid specification of Mchirp_type; can be 'simulated' or 'samples'")
        fkern = eventdict[key]['kde']
        if z_adj is not None:
            if z_adj == 'known':
                z = eventdict[key]['z']
                like_i = single_event_likelihood(Rs,Ms,Mkern,fkern,fprior,Mprior,z,bootstrap=bootstrap)
            else:
                raise TypeError("Invalid specification of z_adj. Only 'known' is currently supported.")
        else:
            like_i = single_event_likelihood(Rs,Ms,Mkern,fkern,fprior,Mprior,z=None,bootstrap=bootstrap)
        likes_all.append(like_i)
    return likes_all   
    
## SECTION 6: POSTERIORS

def get_posterior(Rs,likeR,Rprior):
    '''
    Function to compute the final posterior p(R16|data) given a likelihood and prior.
    
    Arguments:
        Rs (array) : R values at which to compute posterior.
        likeR (array) : Likelihood p(data|R16)
        Rprior (kernel) : R_1.6 prior KDE
        
    Returns:
        R_post (array) : Normalized posterior distribution p(R16|data) evaluated for each R in Rs
    '''
    R_post = likeR*Rprior.pdf(Rs)
    R_post = R_post/np.sum(R_post)
    return R_post
    
## SECTION 7: PLOTTING AND POST-PROCESSING

def plot_aggregate_posterior(Rs,likelihood_list,Rprior,diagnostics=False,Rtrue=None,title=None,
                             legend_anchor=None,saveto=None,include_uniform=False,glow=True,ax=None,Rticks=None,showplot=True):
    '''
    Function to plot the aggregate posterior given a list of event likelihoods and a prior for R_1.6.
    If ax is specified, reroutes to the axis-specific version (plot_aggregate_posterior_on_ax, below).
    
    Arguments:
        Rs (array) : R values at which to compute and plot posterior.
        likelihood_list (list of arrays) : List containing the likelihood p(data_i|R16) for each event.
        Rprior (kernel) : R_1.6 prior KDE
        diagnostics (bool) : Whether to also create diagnostic plot showing posterior-prior difference.
        Rtrue (float) : Injected value of R_1.6, if any
        title (str) : Plot title
        legend_anchor (tuple) : Legend placement. Passed to pyplot's legend bbox_to_anchor.
        saveto (str) : '/path/to/save/figure.pdf' 
                       Note that figure.pdf will have 'post_' or 'diff_' added to beginning depending on use case.
        include_uniform (bool) : Whether to also use a uniform prior.
        glow (bool) : Whether to use mplcyberpunk glow effects.
        ax (axis) : Matplotlib figure axis to plot on. Will call plot_aggregate_posterior_on_ax() (below). 
                    Note that when plotting on a specific axis, the plot will not be saved.
        Rticks (list) : Allows you to specify the x-axis ticks if desired.
    
    Returns:
        Plot.   
    '''
    if ax is not None:
        plot_aggregate_posterior_on_ax(Rs,likelihood_list,Rprior,ax,Rtrue=Rtrue,title=title,legend_anchor=legend_anchor,
                                       include_uniform=include_uniform,glow=glow,Rticks=Rticks)
        return
    else:
        pR_tot = get_aggregate_likelihood(likelihood_list)
        if include_uniform==True:
            if Rtrue is not None:
                plt.axvline(Rtrue,ls='--',color='black',label='True $\mathrm{R_{1.6}}$')
            plt.plot(Rs,Rprior,pdf(Rs)/np.sum(Rprior.pdf(Rs)),label='MM Prior')
            plt.plot(Rs,st.uniform(loc=0,scale=20).pdf(Rs)/np.sum(st.uniform(loc=0,scale=20).pdf(Rs)),
                 label='Uniform Prior')
            plt.plot(Rs,get_posterior(Rs,pR_tot,Rprior),
                     label='Posterior with MM Prior')
            plt.plot(Rs,get_posterior(Rs,pR_tot,st.uniform(loc=0,scale=20)),
                     label='Posterior with Uniform Prior')
            plt.ylim(0,)
        else:
            if Rtrue is not None:
                ymax = np.max([np.max(Rprior.pdf(Rs)/np.sum(Rprior.pdf(Rs))),np.max(get_posterior(Rs,pR_tot,Rprior))])
                plt.ylim(0,1.1*ymax)
                plt.axvline(Rtrue,ls='--',color='black',label='True $\mathrm{R_{1.6}}$')
            plt.plot(Rs,Rprior.pdf(Rs)/np.sum(Rprior.pdf(Rs)),label='$\mathrm{R_{1.6}}$ Prior')
            plt.plot(Rs,get_posterior(Rs,pR_tot,Rprior), label='$\mathrm{R_{1.6}}$ Posterior')
            if Rticks is not None:
                plt.gca().set_xticks(Rticks)

        if legend_anchor is not None:
            plt.legend(bbox_to_anchor=legend_anchor)
        else:
            plt.legend()
        if glow==True:
            mplcyberpunk.add_glow_effects()
        plt.xlabel('$\mathrm{R_{1.6}}$ (km)')
        if title is not None:
            plt.title(title)
        plt.gca().set_yticks([])
        if saveto is not None:
            postname = saveto.split('/')[-1]
            postpath = saveto.replace(postname,'')
            plt.savefig(postpath+'post_'+postname,bbox_inches='tight')
        if showplot==True:
            plt.show()
        if diagnostics==True:
            print("This compares the posterior to the prior. Mostly useful if you expect them to be identical!")
            plt.figure()
            plt.title('Posterior-Prior Consistency')
            plt.plot(Rs,get_posterior(Rs,pR_tot,Rprior) - Rprior.pdf(Rs)/np.sum(Rprior.pdf(Rs)),label='With MM Prior')
            plt.plot(Rs,get_posterior(Rs,pR_tot,st.uniform(loc=0,scale=20)) 
                     - st.uniform(loc=0,scale=20).pdf(Rs)/np.sum(st.uniform(loc=0,scale=20).pdf(Rs)),
                    label='With Uniform Prior')
            plt.legend()
            plt.xlabel('$\mathrm{R_{1.6}}$ (km)')
            plt.ylabel('$p(\mathrm{R_{1.6}}|D) - \pi(\mathrm{R_{1.6}})$')
            if saveto is not None:
                diffname = saveto.split('/')[-1]
                diffpath = saveto.replace(diffname,'')
                plt.savefig(diffpath+'diff_'+diffname,bbox_inches='tight')
            if showplot==True:
                plt.show()
            else:
                plt.close()
        return

def plot_aggregate_posterior_on_ax(Rs,likelihood_list,Rprior,ax,Rtrue=None,title=None,Rticks=None,
                             legend='yes',legend_anchor=None,legend_loc=None,include_uniform=False,glow=True):
    '''
    Creates the aggregate_posterior plot on a specified matplotlib axis. Useful for multipanel plots.
    See plot_aggregate_posterior() (above) for argument details.
    '''
    pR_tot = get_aggregate_likelihood(likelihood_list)
    if include_uniform==True:
        if Rtrue is not None:
            ax.axvline(Rtrue,ls='--',color='black',label='True $\mathrm{R_{1.6}}$')
        ax.plot(Rs,Rprior.pdf(Rs)/np.sum(Rprior.pdf(Rs)),label='MM Prior')
        ax.plot(Rs,st.uniform(loc=0,scale=20).pdf(Rs)/np.sum(st.uniform(loc=0,scale=20).pdf(Rs)),
             label='Uniform Prior')
        ax.plot(Rs,get_posterior(Rs,pR_tot,Rprior),
                 label='Posterior with MM Prior')
        ax.plot(Rs,get_posterior(Rs,pR_tot,st.uniform(loc=0,scale=20)),
                 label='Posterior with Uniform Prior')
        ax.set_ylim(0,)
    else:
        if Rtrue is not None:
            ymax = np.max([np.max(Rprior.pdf(Rs)/np.sum(Rprior.pdf(Rs))),np.max(get_posterior(Rs,pR_tot,Rprior))])
            ax.set_ylim(0,1.1*ymax)
            ax.axvline(Rtrue,ls='--',color='black',label='True $\mathrm{R_{1.6}}$')
        ax.plot(Rs,Rprior.pdf(Rs)/np.sum(Rprior.pdf(Rs)),label='$\mathrm{R_{1.6}}$ Prior')
        ax.plot(Rs,get_posterior(Rs,pR_tot,Rprior), label='$\mathrm{R_{1.6}}$ Posterior')
        if Rticks is not None:
            ax.set_xticks(Rticks)

    if legend=='yes':
        if legend_anchor is not None:
            ax.legend(bbox_to_anchor=legend_anchor)
        elif legend_loc is not None:
            ax.legend(loc=legend_loc)
        else:
            ax.legend()
    if glow==True:
        mplcyberpunk.add_glow_effects(ax)
    ax.set_xlabel('$\mathrm{R_{1.6}}$ (km)')
    if title is not None:
        ax.set_title(title)
    ax.set_yticks([])

    return

def plot_posterior_by_Nevents(Rs,likelihood_list,Rprior,Ns,ax=None,Rtrue=None,title=None,
                              legend_anchor=None,legend_loc=None,glow=False,saveto=None,showplot=True):
    '''
    Function to plot the evolution of the posterior by event.
    
    Arguments:
        Rs (array) : R values at which to compute and plot posterior.
        likelihood_list (list of arrays) : List containing the likelihood p(data_i|R16) for each event.
        Rprior (kernel) : R_1.6 prior KDE
        diagnostics (bool) : Whether to also create diagnostic plot showing posterior-prior difference.
        Ns (list) : Which N_events to plot the posterior for.
        ax (matplotlib Axes) : Axes object on which to plot if specified.
        Rtrue (float) : Injected value of R_1.6, if any
        title (str) : Plot title
        legend_anchor (tuple) : Legend placement. Passed to pyplot's legend bbox_to_anchor.
        legend_loc (str) : Alternate legend placement, passed to pyplot's legend loc.
        glow (bool) : Whether to use mplcyberpunk glow effects.
        saveto (str) : '/path/to/save/figure.pdf'
    
    Returns:
        Plot
    '''
    if ax is not None:
        ax = ax
    else:
        plt.figure()
        ax = plt.gca()
    pR_tots = []
    pR_maxs = []
    for N in Ns:
        pR_tot_N = get_posterior(Rs,get_aggregate_likelihood(likelihood_list[:N]),Rprior)
        pR_tots.append(pR_tot_N)
        pR_maxs.append(np.max(pR_tot_N))

    ax.plot(Rs,Rprior.pdf(Rs)/np.sum(Rprior.pdf(Rs)),label='$\mathrm{R_{1.6}}$ Prior',color='k')
    ymax = np.max([np.max(Rprior.pdf(Rs)/np.sum(Rprior.pdf(Rs))),np.max(pR_maxs)])
    ax.set_ylim(0,1.1*ymax)
    for N, pR_tot_N in zip(Ns,pR_tots):
        ax.plot(Rs,pR_tot_N,label='$\mathrm{R_{1.6}}$ Posterior ($\mathrm{N}_{det}$'+'={})'.format(N)) 
    if Rtrue is not None:
        ax.axvline(Rtrue,ls='--',color='k',label='True $\mathrm{R_{1.6}}$')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('$\mathrm{R_{1.6}}$ Posterior by $N_{det}$')
    ax.set_xlabel('$\mathrm{R_{1.6}}$ (km)')
    ax.set_yticks([])
    if legend_anchor is not None:
        ax.legend(bbox_to_anchor=legend_anchor)
    elif legend_loc is not None:
        ax.legend(loc=legend_loc)
    else:
        ax.legend()
    if glow==True:
        mplcyberpunk.add_glow_effects(ax)
    if ax is None:
        if saveto is not None:
                plt.savefig(saveto,bbox_inches='tight')
        if showplot==True:
            plt.show()
        else:
            plt.close()

    
def multiplot_3panel_post(Rs,dict_dict,Rprior,sensitivities=['2x','2.5x','3x'],eoss=['sly4','sfhx41','dd2'],
                          Rtruevals=[11.54,11.98,13.26],title=None,saveto=None,legend_anchor=None,glow=False,
                         legend_loc=None,obs_run=False,nosignal=False,dist_cut=None,dict_type='loaded',legend_num='all'):
    '''
    Function to create three-panel plots to compare posteriors across different sensitivities/EoSs.
    I tried to write it generally, but give no guarantees of functionality outside of its intended use case (That is,
        the Soultanis sims injected via BayesWave at certain sensitivities)
    See plot_aggregate_posterior() for documentation of the plotting arguments.
    
    Arguments:
        (arguments for plot_aggregate_posterior()) : see plot_aggregate_posterior() for documentation
        dict_dict (dict) : Dictionary containing a posterior_eventdict (as generated by save_posterior_eventdict()) for
                           each sensitvity-EoS combo
        sensitivites (list of str) : List of sensitivities used.
        eoss (list of str) : List of EoSs used.
        Rtruevals (list of float) : List of corresponding R_1.6 values for EoSs used
        nosignal (bool) : Whether real signals were analyzed or not.
        dist_cut (float) : Only compute aggregate posterior for events with distance < dist_cut (in Mpc)
        dict_type (str) : 'loaded' or 'raw'. 'loaded' assumes dict_dict is of the form generated by save_posterior_eventdict()
                          and load_all_pickles_in_jar(). 'raw' assumes a dictionary of standard posterior_eventdicts as output
                          by hbpm_analysis.run_analysis() whose keys are the same as the passed sensitivities. 
                          Note: 'raw' only supports single-EoS plotting.
        legend_num (str) : 'all' or 'one'. Whether to place a legend on each subplot or only the upper right one.
    
    Returns:
        Plot.
    '''
    if len(dict_dict) != len(sensitivities)*len(eoss):
        raise TypeError("Need one dictionary per EoS per sensitivity!")
    if len(eoss) != len(Rtruevals):
        raise TypeError("Need true value of R_1.6 for every EoS and vice versa!")
    ## set shared keys that are not related to individual events
    nonunique_keys = ['eos','Rs','prior']
    fig, axes = plt.subplots(1,len(sensitivities),figsize=(15,4))
    ## set EoS colors
    if len(eoss) > 1:
        eos_colors = ['mediumorchid','teal','goldenrod']
        prior_color = 'slategray'
    elif len(eoss)==1:
        eos_colors = ['teal']
        prior_color = 'mediumorchid'
    else:
        raise TypeError("Empty EoS Array?")
    for ax, sens in zip(axes.flatten(),sensitivities):
        ax.set_yticks([])
        if Rtruevals is not None:
            for Rtrue, eos in zip(Rtruevals,eoss):
                trans = ax.get_xaxis_transform()
                if eos==eoss[-1]:
                    ax.axvline(Rtrue,ls='--',color='black',label='True $\mathrm{R_{1.6}}$')
                else:
                    ax.axvline(Rtrue,ls='--',color='black')
                if len(eoss)>1:
                    plt.text(Rtrue, .8, eos.replace('sfhx41','SFHX'), transform=trans, rotation= 60)
        ## prior
        ax.plot(Rs,Rprior(Rs)/np.sum(Rprior(Rs)),label='$\mathrm{R_{1.6}}$ Prior',color=prior_color)
        ## posteriors
        ymaxlist = [np.max(Rprior(Rs)/np.sum(Rprior(Rs)))]
        
        if dict_type=='loaded':
            for eos, ec in zip(eoss,eos_colors):
                if obs_run==True:
                    if nosignal==True:
                        key_ij = sens+'_sim_withscaling_nosignal'
                    else:
                        key_ij = sens+'_sim_withscaling_updated'
                else:
                    key_ij = eos+'_'+sens
                dict_ij = dict_dict[key_ij]
                if len(eoss)==1:
                    label_ij = '$\mathrm{R_{1.6}}$ Posterior'
                else:
                    label_ij = eos.replace('sfhx','SFHX').replace('41','')
                likes_ij = [dict_ij['post_dict'][key]['likelihood_i'] for key in dict_ij['post_dict'].keys()
                            if key not in ['eos','Rs','prior']]
                if dist_cut is not None:
                    nearby_filt = [(float(key.split('_')[-1].replace('-2','')) < dist_cut) 
                                   for key in dict_ij['post_dict'].keys() if key not in ['eos','Rs','prior']]
                    likes_ij = [like for filt,like in zip(nearby_filt,likes_ij) if filt==True]
                post_ij = get_posterior(Rs,get_aggregate_likelihood(likes_ij),Rprior)
                ax.plot(Rs,post_ij,label=label_ij,color=ec)
                ymaxlist.append(np.max(post_ij))


            ax.set_xlabel('$\mathrm{R_{1.6}}$ (km)')
            if legend_num=='all':
                if legend_anchor is not None:
                    ax.legend(bbox_to_anchor=legend_anchor)
                elif legend_loc is not None:
                    ax.legend(loc=legend_loc)
                else:
                    ax.legend()
            elif legend_num=='one':
                if sens==sensitivities[-1]:
                    if legend_anchor is not None:
                        ax.legend(bbox_to_anchor=legend_anchor)
                    elif legend_loc is not None:
                        ax.legend(loc=legend_loc)
                    else:
                        ax.legend()
            if 'x' in sens:
                factor = sens.replace('x','\\times')
                ax.set_title('${}$A+ Sensitivity'.format(factor))
            elif 'O' in sens:
                if dist_cut is not None:
                    ax.set_title('{} ($d_L <$ {:0.0f} Mpc)'.format(sens,dist_cut).replace('O4O5','O4+O5'))
                else:
                    ax.set_title('{} (All Events)'.format(sens).replace('O4O5','O4+O5'))

            if glow==True:
                ymax = np.max(ymaxlist)
                ax.set_ylim(0,1.1*ymax)
                mplcyberpunk.add_glow_effects(ax)
        elif dict_type=='raw':
            if len(eoss)>1:
                print("Please get dict via save_posterior_eventdict() and load_all_pickles_in_jar().")
                raise TypeError("Multiple Equations of State not supported for raw input.")
            
            key_ij = sens
            dict_ij = dict_dict[key_ij]
            label_ij = '$\mathrm{R_{1.6}}$ Posterior'
            likes_ij = [dict_ij[key]['likelihood_i'] for key in dict_ij.keys() if key not in ['eos','Rs','prior']]
            if dist_cut is not None:
                nearby_filt = [(float(key.split('_')[-1].replace('-2','')) < dist_cut) 
                               for key in dict_ij.keys() if key not in ['eos','Rs','prior']]
                likes_ij = [like for filt,like in zip(nearby_filt,likes_ij) if filt==True]
            post_ij = get_posterior(Rs,get_aggregate_likelihood(likes_ij),Rprior)
            ax.plot(Rs,post_ij,label=label_ij,color=eos_colors[0])
            ymaxlist.append(np.max(post_ij))


            ax.set_xlabel('$\mathrm{R_{1.6}}$ (km)')
            if legend_num=='all':
                if legend_anchor is not None:
                    ax.legend(bbox_to_anchor=legend_anchor)
                elif legend_loc is not None:
                    ax.legend(loc=legend_loc)
                else:
                    ax.legend()
            elif legend_num=='one':
                if sens==sensitivities[-1]:
                    if legend_anchor is not None:
                        ax.legend(bbox_to_anchor=legend_anchor)
                    elif legend_loc is not None:
                        ax.legend(loc=legend_loc)
                    else:
                        ax.legend()
            if 'x' in sens:
                factor = sens.replace('x','\\times')
                ax.set_title('${}$A+ Sensitivity'.format(factor))
            elif 'O' in sens:
                if dist_cut is not None:
                    ax.set_title('{} ($d_L <$ {:0.0f} Mpc)'.format(sens,dist_cut).replace('O4O5','O4+O5'))
                else:
                    ax.set_title('{}'.format(sens).replace('O4O5','O4+O5'))

            if glow==True:
                ymax = np.max(ymaxlist)
                ax.set_ylim(0,1.1*ymax)
                mplcyberpunk.add_glow_effects(ax)
    if saveto is not None:
        plt.savefig(saveto,bbox_inches='tight')
    plt.show()
    return
    
def multiplot_3panel_scaling(Rs,dict_dict,Rprior,sensitivities=['2x','2.5x','3x'],eoss=['sly4','sfhx41','dd2'],
                             Rtruevals=[11.54,11.98,13.26],Ns=[40,80,120,132],title=None,saveto=None,
                             legend_anchor=None,glow=False,legend_loc=None):
    '''
    Function to create three-panel plots to compare posterior scaling by N_events across different sensitivities/EoSs.
    I tried to write it generally, but give no guarantees of functionality outside of its intended use case (That is,
        the Soultanis sims injected via BayesWave at certain sensitivities)
    See plot_aggregate_posterior() for documentation of the plotting arguments.
    
    Arguments:
        (arguments for plot_aggregate_posterior()) : see plot_aggregate_posterior() for documentation
        dict_dict (dict) : Dictionary containing a posterior_eventdict (as generated by save_posterior_eventdict()) for
                           each sensitvity-EoS combo
        sensitivites (list of str) : List of sensitivities used.
        eoss (list of str) : List of EoSs used.
        Rtruevals (list of float) : List of corresponding R_1.6 values for EoSs used
        Ns (list of int) : N_event thresholds at which to plot the cumulative posterior.
    
    Returns:
        Plot.
    '''
    if len(dict_dict) != len(sensitivities)*len(eoss):
        raise TypeError("Need one dictionary per EoS per sensitivity!")
    if len(eoss) != len(Rtruevals):
        raise TypeError("Need true vale of R_1.6 for every EoS and vice versa!")
    ## set shared keys that are not related to individual events
    nonunique_keys = ['eos','Rs','prior']
    fig, axes = plt.subplots(1,len(sensitivities),figsize=(15,4))
    ## set EoS colors
    eos_colors = ['mediumorchid','teal','goldenrod']
    prior_color = 'slategray'
    for ax, sens in zip(axes.flatten(),sensitivities):
        ax.set_yticks([])
        if Rtruevals is not None:
            for Rtrue, eos in zip(Rtruevals,eoss):
                trans = ax.get_xaxis_transform()
                if eos==eoss[-1]:
                    ax.axvline(Rtrue,ls='--',color='black',label='True $\mathrm{R_{1.6}}$')
                else:
                    ax.axvline(Rtrue,ls='--',color='black')
                if len(eoss)>1:
                    plt.text(Rtrue, .8, eos.replace('sfhx41','SFHX'), transform=trans, rotation= 60)
        ## prior
        ax.plot(Rs,Rprior(Rs)/np.sum(Rprior(Rs)),label='$\mathrm{R_{1.6}}$ Prior',color=prior_color)
        ## posteriors
        ymaxlist = [np.max(Rprior(Rs)/np.sum(Rprior(Rs)))]
        for eos, ec in zip(eoss,eos_colors):
            key_ij = eos+'_'+sens
            dict_ij = dict_dict[key_ij]
            likes_ij = [dict_ij['post_dict'][key]['likelihood_i'] for key in dict_ij['post_dict'].keys()
                        if key not in ['eos','Rs','prior']]
            pR_tots = []
            pR_maxs = []
            for N in Ns:
                pR_tot_N = get_posterior(Rs,get_aggregate_likelihood(likes_ij[:N]),Rprior)
                pR_tots.append(pR_tot_N)
                pR_maxs.append(np.max(pR_tot_N))
            ymax_ij = np.max([np.max(Rprior.pdf(Rs)/np.sum(Rprior.pdf(Rs))),np.max(pR_maxs)])
            for N, pR_tot_N in zip(Ns,pR_tots):
                ax.plot(Rs,pR_tot_N,label='$\mathrm{N}_{det}$'+'={}'.format(N))
            ymaxlist.append(ymax_ij)
        
        
        ax.set_xlabel('$\mathrm{R_{1.6}}$ (km)')
        if legend_anchor is not None:
            ax.legend(bbox_to_anchor=legend_anchor)
        elif legend_loc is not None:
            ax.legend(loc=legend_loc)
        else:
            ax.legend()
        if 'x' in sens:
            factor = sens.replace('x','\\times')
            ax.set_title('${}$A+ Sensitivity'.format(factor))
        elif 'O' in sens:
            ax.set_title('{} Sensitivity'.format(sens))
        
        if glow==True:
            ymax = np.max(ymaxlist)
            ax.set_ylim(0,1.1*ymax)
            mplcyberpunk.add_glow_effects(ax)
    if saveto is not None:
        plt.savefig(saveto,bbox_inches='tight')
    plt.show()
    return    
    

def get_post_stats(post_dist,Rs,bounds=(0.025,0.975),verbose=True,latex=False):
    '''
    Function to get the posterior mean and confidence interval (default 95%).
    
    Arguments:
        post_dist (array) : R_1.6 posterior distribution
        Rs (array) : R values at which post_dist is evaluated
        bounds (tuple) : (lower limit,upper limit) of confidence interval. Default gives 95% C.I..
        verbose (bool) : Whether to print results.
        latex (bool) : If printing results, whether to output as an easily copied LaTeX string.
        
    Returns:
        mean (float) : Posterior mean
        lower (float) : Lower confidence interval bound
        upper (float) : Upper confidence interval bound
    '''
    ## check that posterior is normalized; if not, normalize it.
    if np.sum(post_dist) != 1:
        post_dist = post_dist/np.sum(post_dist)
    ## generally my grid resolution won't be sufficient for precise bounds, so upsample with an interpolator
    post_interp = interp1d(Rs,post_dist)
    Rs_new = np.linspace(Rs.min(),Rs.max(),20000)
    ## renormalize
    post_dist = post_interp(Rs_new)/np.sum(post_interp(Rs_new))
    mean = np.sum(Rs_new*post_dist)
    idr = np.arange(len(post_dist))
    cp_idr = [np.sum(post_dist[:i+1]) for i in idr]
    lower_idr = np.argmin(np.abs(np.array(cp_idr) - bounds[0]))
    upper_idr = np.argmin(np.abs(np.array(cp_idr) - bounds[1]))
    lower = Rs_new[lower_idr]
    upper = Rs_new[upper_idr]
    if verbose==True:
        if latex:
            print("$\R={:0.2f}^".format(mean)+"{+"+"{:0.2f}".format(upper-mean)+"}_{-"+"{:0.2f}".format(mean-lower)+"}$")
        else:
            print("R_1.6 = {:0.2f} (+{:0.2f},-{:0.2f}) km".format(mean,upper - mean,mean-lower,))
    return mean, lower, upper

def save_posterior_eventdict(events,eventdict,ev_df,likelihoods,Rprior,Rrange,saveto,event_type,Rtrue=None,returndict=False):
    '''
    Function to save posterior along with all necessary data to recreate it. Stores to a .pickle file.
    
    Arguments:
        events (list) : List of events. If using NR sims, this is the list of NR file names in standard Soultanis format.
                        If simulated data, use list of chirp masses.
        eventdict (dict) : Event dictionary as created by the gen_*_eventdict() functions above.
        ev_df (DataFrame) : Dataframe with information about the injected signals. Only needed for the NR waveform name.
                            Note: set to None for simulated datasets
        likelihoods (list of arrays) : List of individual event likelihoods, in eventdict order.
        Rprior (array) : R_1.6 prior samples used.
        Rrange (array) : R values at which likelihoods have been evaluated
        saveto (str) : '/path/where/file/will/be/saved.pickle'
        event_type (str) : How the events were made. Can be 'BayesWave' or 'simulated'.
        Rtrue (float) : Injected value of R_1.6. Only needed for simulated data.
        returndict (bool) : Whether to also return the created dictionary.
        
    Returns:
        Saved .pickle file
        (optional) post_dict (dict) : If returndict=True, the unpickled dictionary.
    '''
    post_dict = {}
    if event_type=='BayesWave':
        post_dict['eos'] = events[0].split('/')[-1].split('_')[0]
    elif event_type=='simulated':
        post_dict['eos'] = 'Simulated data; injected R_1.6={}'.format(Rtrue)
    else:
        raise TypeError('Error: invalid event_type specified.')
    post_dict['Rs'] = Rrange
    post_dict['prior'] = Rprior
    ## store individual event likelihoods with accompanying metadata
    for mstr, event, likelihood in zip([*eventdict.keys()],events,likelihoods):
#         mstr = event.split('/')[-1].split('_')[1] ##soultanis-style event key e.g. m1=1.01,m2=1.34 -> 101134
        post_dict[mstr] = eventdict[mstr]
        if 'kde' in post_dict[mstr].keys():
            post_dict[mstr].pop('kde')
        post_dict[mstr]['likelihood_i'] = likelihood
        if event_type=='BayesWave':
            waveform = ev_df[parse_datafile(ev_df['file']) == event.split('/')[-1].split('_')[1]]['file'].item()
            post_dict[mstr]['waveform'] = waveform
    if '.pickle' not in saveto:
        saveto=saveto+'.pickle'
    with open(saveto, 'wb') as outloc:
        pickle.dump(post_dict,outloc)
    print("Saved file to {}".format(saveto))
    if returndict==True:
        return post_dict
    else:
        return
            
def load_all_pickles_in_jar(datadir):
    '''
    Utility function to load all .pickle files in a specified directory. 
    Note: intended for use with specific simulations, may need to be extended later.
    
    Arguments:
        datadir (str) : '/path/to/directory/with/all/pickle/files/'
    
    Returns:
        dict_of_dicts (dict) : Dictionary containing a dictionary for each pickle file in the specified directory + metadata.
    '''
    ## I think I'm funny
    files = glob(datadir+'/*.pickle')
    dict_of_dicts = {}
    for file in files:
        with open(file, 'rb') as fp:
            post_dict_i = pickle.load(fp)
        key_i = file.split('/')[-1].replace('.pickle','').replace('_all','')
        if ('O4' not in key_i) and ('O5' not in key_i):
            eos_i = key_i.split('_')[0].replace('sfhx','SFHX')
            sens_i = key_i.split('_')[1]
        elif 'O4O5_sim' in key_i:
            eos_i = 'SFHX'
            sens_i = 'O4+O5'
        elif 'O4_sim' in key_i:
            eos_i = 'SFHX'
            sens_i = 'O4'
        elif 'O5_sim' in key_i:
            eos_i = 'SFHX'
            sens_i = 'O5'
        else:
            raise TypeError('Unrecognized file format/sensitivity!')
        dict_of_dicts[key_i] = {}
        dict_of_dicts[key_i]['eos'] = eos_i
        dict_of_dicts[key_i]['sensitivity'] = sens_i
        dict_of_dicts[key_i]['post_dict'] = post_dict_i
    return dict_of_dicts

def load_posterior_pickle(post_pickle):
    '''
    Utility function to load all .pickle files in a specified directory. 
    Note: intended for use with specific simulations, may need to be extended later.
    
    Arguments:
        post_pickle (str) : '/path/to/pickle/file.pickle'
    
    Returns:
        post_dict (dict) : Unpickled posterior eventdict
    '''

    with open(post_pickle, 'rb') as fp:
        post_dict = pickle.load(fp)
        
    return post_dict

def unpack_posterior_dict(posterior_eventdict):
    '''
    Utility function to get R array and corresponding likelihood_list/prior from a posterior_eventdict.
    
    Arguments:
        posterior_eventdict (dict) : posterior event dictionary as produced by save_posterior_eventdict() or load_posterior_pickle().
    
    Returns:
        Rs (array) : R values at which the likelihoods are evaluated.
        Rprior_kernel (scipy.stats.gaussian_kde) : R_1.6 prior KDE
        likelihood_list (list of arrays) : List of individual event likelihoods
    '''
    
    likelihood_list = [posterior_eventdict[key]['likelihood_i'] for key in posterior_eventdict.keys() if key not in ['eos','Rs','prior']]
    if type(posterior_eventdict['prior']) is str:
        Rprior_kernel = st.uniform(loc=9,scale=6)
    else:
        Rprior_kernel = kde(posterior_eventdict['prior'])
    Rs = posterior_eventdict['Rs']
    
    return Rs, Rprior_kernel, likelihood_list

## SECTION 8: LOADING AND PLOTTING BAYESWAVE RESULTS
def BW_power2strain(fs,powers):
    '''
    Function to convert from BayesWave output "power" to strain (2*abs(h)*sqrt(f),sqrt(S_n))
    
    Arguments:
        fs (array) : Frequencies.
        powers (array) : Corresponding powers.
    
    Returns:
        strains (array) : Converted strains.
    '''
    
    return 2*np.sqrt(np.abs(powers))#*np.sqrt(fs))
#     return 2*np.sqrt(powers) ## do NOT ask me why this is what works, but it gets the noise PSDs to match up

def get_waveform(filename):
    '''
    Function to load waveform reconstruction data from a Bayeswave output. 
    
    Arguments:
        filename (str) : '/path/to/BayesWave/reconstruction/data/signal_median_PSD_H1.dat'
    
    Returns:
        samples (array) : All sampled waveforms.
        median_waveform (array) : Median reconstructed waveform
        50low (array) : Lower 50% C.I. of waveform reconstructions
        50high (array) : Upper 50% C.I. of waveform reconstructions
        90low (array) : Lower 90% C.I. of waveform reconstructions
        90high (array) : Upper 90% C.I. of waveform reconstructions
    '''
    names = ['samples','median_waveform','50low','50high','90low','90high']
    data = np.recfromtxt(filename,names=names)
    return (data['samples'],data['median_waveform'],data['50low'],data['50high'],data['90low'],data['90high'])

def get_inj(filename):
    '''
    Simple wrapper function to get a BayesWave injection spectrum.
    
    Arguments:
        filename (str) : '/path/to/BayesWave/injection/data/injected_whitened_spectrum_H1.dat'
        
    Returns:
        injected_spectrum (array) : Injected waveform spectrum.
    '''
    injected_spectrum = np.genfromtxt(filename)
    return injected_spectrum

def get_reconstruction_data(topdir,psd_name=None,powerspec_name=None,inj_name=None):
    '''
    Wrapper function to get all needed datasets for making the BayesWave plots below. Assumes all files are in the same directory.
    
    Arguments:
        topdir (str) : '/path/to/directory/with/files/'
        psd_name (str) : (optional) name of noise PSD file
        powerspec_name (str) : (optional) name of reconstruction power spectrum file
        inj_name (str) : (optional) name of injection file
    
    Returns:
        powerspec_info (tuple) : reconstructed signal spectrum data
        psd_info (tuple) : instrumental noise power spectrum data
        injected_spectrum (array) : injected signal data
    '''
    if psd_name is not None:
        psd_info = get_waveform(topdir+'/'+psd_name)
    else:
        psd_info = get_waveform(topdir+'/signal_median_PSD_H1.dat')
    if powerspec_name is not None:
        powerspec_info = get_waveform(topdir+'/'+powerspec_name)
    else:
        powerspec_info = get_waveform(topdir+'/signal_median_frequency_domain_waveform_spectrum_H1.dat')
    if inj_name is not None:
        injected_spectrum = get_inj(topdir+'/'+inj_name)
    else:
        injected_spectrum = get_inj(topdir+'/injected_whitened_spectrum_H1.dat')
    
    return powerspec_info, psd_info, injected_spectrum

def plot_reconstruction_spec(powerspec_info,psd_info,injected_spectrum,y_units='power',figsize=None,xlim=(1.5e3,4e3),ylim=None,
                             ylabel=None,title="BayesWave Reconstructed Power Spectra",saveto=None,**legend_kwargs):
    '''
    Function to plot a BayesWave signal reconstruction.
    
    Arguments:
        powerspec_info (tuple) : reconstructed power spectrum data; see get_reconstruction_data() above
        psd_info (tuple) : instrumental noise power spectrum data; see get_reconstruction_data() above
        injected_spectrum (array) : injected signal data; see get_reconstruction_data() above
        y_units (str) : units to use for the y axis. Can be 'strain' (2*|h(f)|sqrt(f) & sqrt(S_n)) or 'power' (raw BayesWave output)
        figsize (tuple) : matplotlib figsize if desired
        xlim (tuple) : matplotlib xlim if desired
        ylim (tuple) : matplotlib ylim if desired
        ylabel (str) : matplotlib ylabel if desired
        title (str) : matplotlib title if desired
        saveto (str) : '/path/to/save/location.png'
        **legend_kwargs : Any legend keyword arguments you care to pass.
    
    Returns:
        Plot.
    '''
    
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    ax = plt.gca()
    plot_reconstruction_spec_on_ax(ax,powerspec_info,psd_info,injected_spectrum,
                                   y_units=y_units,xlim=xlim,ylim=ylim,xlabel=xlabel,ylabel=ylabel,
                                   title=title,**legend_kwargs)
    if saveto is not None:
        plt.savefig(saveto,bbox_inches='tight')
    plt.show()
    return

def plot_reconstruction_spec_on_ax(ax,powerspec_info,psd_info,injected_spectrum,
                                   y_units='power',xlim=(1.5e3,4e3),ylim=None,xlabel=None,ylabel=None,
                                   title="BayesWave Reconstructed Power Spectra",**legend_kwargs):
    '''
    Function to plot a BayesWave signal reconstruction on a specified matplotlib axis.
    
    Arguments:
        ax (axis) : Matplotlib axes object on which to plot the reconstruction.
        powerspec_info (tuple) : reconstructed power spectrum data; see get_reconstruction_data() above
        psd_info (tuple) : instrumental noise power spectrum data; see get_reconstruction_data() above
        injected_spectrum (array) : injected signal data; see get_reconstruction_data() above
        xlim (tuple) : matplotlib xlim if desired
        ylim (tuple) : matplotlib ylim if desired
        ylabel (str) : matplotlib ylabel if desired
        title (str) : matplotlib title if desired
        **legend_kwargs : Any legend keyword arguments you care to pass.
    
    Returns:
        Plot
    '''
    injcolor = 'teal'
    signalcolor = 'mediumorchid'
    
    if y_units == 'power':
        if ylabel is None:
            ax.set_ylabel('PSD ($\mathrm{Hz}^{-1}$)')
        ## injection
        injspec_freqs = injected_spectrum[:,0]
        injspec = injected_spectrum[:,1]
        ## PSD
        psd_freqs = psd_info[0]
        psd = psd_info[1]
        ## reconstruction
        postspec_freqs = powerspec_info[0]
        postspec_50 = [powerspec_info[2],powerspec_info[3]]
        postspec_90 = [powerspec_info[4],powerspec_info[5]]
        postspec_med = powerspec_info[1]
    elif y_units == 'strain':
        if ylabel is None:
            ax.set_ylabel('$2 \mid h(f) \mid \sqrt{f}$ & $\sqrt{S_n}$')
        ## injection
        injspec_freqs = injected_spectrum[:,0]
        injspec = BW_power2strain(injspec_freqs,injected_spectrum[:,1])*2*np.sqrt(injspec_freqs)
#                   2*np.sqrt(np.abs(injected_spectrum[:,1])*np.sqrt(injspec_freqs)
        ## Detector PSD
        psd_freqs = psd_info[0]
        psd = BW_power2strain(psd_info[0],psd_info[1])
        ## reconstruction
        postspec_freqs = powerspec_info[0]
        postspec_50 = [BW_power2strain(postspec_freqs,powerspec_info[2])*2*np.sqrt(postspec_freqs),
                       BW_power2strain(postspec_freqs,powerspec_info[3])*2*np.sqrt(postspec_freqs)]
        postspec_90 = [BW_power2strain(postspec_freqs,powerspec_info[4])*2*np.sqrt(postspec_freqs),
                       BW_power2strain(postspec_freqs,powerspec_info[5])*2*np.sqrt(postspec_freqs)]
        postspec_med = BW_power2strain(postspec_freqs,powerspec_info[1])*2*np.sqrt(postspec_freqs)
    else:
        raise TypeError("Invalid units specified. Only 'strain' and 'power' are supported.")
    # plot injection
    ax.semilogy(injspec_freqs,injspec,injcolor, linewidth=1,label='Injection')

    # plot psd
#     ax.fill_between(psd_info[0],psd_info[4],psd_info[5],color='grey',alpha=0.8)
    ax.semilogy(psd_freqs,psd,color='k',ls='-',label='Detector PSD')

    # plot powerspec
    ax.fill_between(postspec_freqs,postspec_50[0],postspec_50[1],color=signalcolor,alpha=0.5)
    ax.fill_between(postspec_freqs,postspec_90[0],postspec_90[1],color=signalcolor,alpha=0.3)
    ax.plot(postspec_freqs,postspec_med,color=signalcolor,label='Reconstruction')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    
    ax.legend(**legend_kwargs)
    
    return


def plot_reconstruction_with_fpeak(powerspec_info,psd_info,injected_spectrum,fpeak_post,
                                   xlim=(1.5e3,4e3),recon_ylim=None,bins=100,saveto=None,xticks=None,
                                   xlabel='Frequency (kHz)',suptitle=None,**legend_kwargs):
    '''
    Function to make a stacked plot of a BayesWave post-merger signal reconstruction with corresponding fpeak posterior samples.
    
    Arguments:
        powerspec_info (tuple) : reconstructed power spectrum data; see get_reconstruction_data() above
        psd_info (tuple) : instrumental noise power spectrum data; see get_reconstruction_data() above
        injected_spectrum (array) : injected signal data; see get_reconstruction_data() above
        fpeak_post (array) : Peak frequency posterior samples
        xlim (tuple) : matplotlib xlim if desired (shared by both plots)
        recon_ylim (tuple) : matplotlib ylim for the reconstruction plot, if desired
        bins (int) : Number of bins to use for the peak frequency posterior histogram
        saveto (str) : '/path/to/save/location.png'
        xticks (list) : matplotlib xticks, if desired
        xlabel (str) : matplotlib xlabel if desired (shared by both plots)
        suptitle (str) : matplotlib suptitle if desired
        **legend_kwargs : Any legend keyword arguments for the reconstruction plot you care to pass.
    '''
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(6,9),gridspec_kw={'height_ratios': [3,1]})
    plot_reconstruction_spec_on_ax(ax1,powerspec_info,psd_info,injected_spectrum,xlim=xlim,ylim=recon_ylim,**legend_kwargs)
    ax2.hist(fpeak_post,bins=bins,color='mediumorchid',alpha=0.8)
    ax2.set_xlabel(xlabel)
    ax2.set_title("$\mathrm{f_{peak}}$ Posterior Samples")
    ax2.set_yticks([])
    if xticks is None:
        ax2.set_xticks(ticks=[1.5e3,2e3,2.5e3,3e3,3.5e3,4e3])
        ax2.set_xticklabels([1.5,2.0,2.5,3.0,3.5,4.0])
    else:
        ax2.set_xticks(ticks=xticks)
    plt.tight_layout()
    if suptitle is not None:
        plt.suptitle(suptitle)
    if saveto is not None:
        plt.savefig(saveto,bbox_inches='tight')
    plt.show()
    return


