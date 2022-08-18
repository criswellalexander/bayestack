#!/usr/bin/env python
# coding: utf-8

'''
This code executes the Hierarchical Bayesian Post-merger analysis for the use case of observing run simulations.
If you want to do something else, use the modular functions in hbpm_utils.py to build a custom analysis.
    - see example jupyter notebooks for reference

Usage: hbpm_analysis.py [datadir] [sim_df_file] [saveto] --optional_arguments

Arguments:

[1] datadir : '/path/to/BayesWave/Fpeak/output/head/directory/'
[2] sim_df_file : '/path/to/csv/with/observing/run/event/details.csv'
[3] saveto : '/path/to/save/directory/'

Optional:
--eos : Name of equation of state as used in BayesWave outputs (e.g. sfhx)
--ev_df_file : '/path/to/csv/with/NR/simulation/details.csv' (e.g. ./nr_files/sfhx_ev_df.csv)
--fprior_file : '/path/to/file/with/fpeak/prior/samples.txt'
--Rprior_file : '/path/to/file/with/R16/prior/samples.txt'
--nosignal : If included, does a signalless run, replacing all BayesWave fpeak posterior draws with draws from the fpeak prior.
--showplots : If included, displays inline plots. (Plots will always be saved to the output directory in any case.)
--N_thresholds : N_event thresholds at which to plot the multi-event posterior for the posterior evolution plot. (e.g. 15 25 35)
--bootstrap : '/path/to/file/with/bootstrapped/empirical/relation/coefficients.txt' (For incorporating empirical relation error.)
'''

import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
import scipy.stats as st
import matplotlib
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde as kde
from scipy.stats.mstats import mquantiles as mq
from pesummary.core.plots.bounded_1d_kde import bounded_1d_kde
from glob import glob
import pandas as pd
import sys 
import os
from tabulate import tabulate
import dill
import argparse
sys.path.append(os.path.abspath('./hbpm_utils/'))
from hbpm_utils import *
matplotlib.rcParams['figure.figsize'] = (8.08, 5.)
matplotlib.rcParams['xtick.labelsize'] = 12.0
matplotlib.rcParams['ytick.labelsize'] = 12.0
matplotlib.rcParams['axes.labelsize'] = 14.0
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["mediumorchid", "teal", "goldenrod","slategray"])



def run_analysis(datadir,fprior_spec,Rprior,eos,sim_df_file,ev_df_file,z_adj=None,Mchirp_scaling='none',
                 Rbounds=None,Ns=None,saveto=None,showplots=True,nosignal=False,Rtrue=None,bootstrap=None,prior_bandwidth=0.15,posterior_bandwidth=0.15,
                 aggregation='sum',ifos='H1,L1',seed=None):
    '''
    This function runs the Hierarchical Bayesian Post-merger analysis end-to-end.
    
    Arguments:
        datadir (str) : '/path/to/directory/with/all/BayesWaveFpeak/outputs/'
        fprior_spec (str) : Either '/path/to/fpeak/prior/samples.txt' or 'uniform'
        Rprior (str) : Either '/path/to/R16/prior/samples.txt' or 'uniform'
        eos (str) : Equation of State as used in the Soultanis sim filenames (e.g., 'sfhx', 'dd2', 'sly4')
        sim_df_file (str) : '/path/to/csv/with/observing/run/simulation/info.csv' (e.g. allsky.dat)
        ev_df_file (str) : '/path/to/csv/with/NR/simulation/details.csv'
        z_adj (str) : Whether to account for redshift. If 'known', uses luminosity distances from ev_df_file 
                      to compute redshift correction for each event, assuming standard LambdaCDM. 
                      ('posteriors' coming soon (TM), which will allow for use of inspiral redshift posteriors.)
        Mchirp_scaling (str): How to scale the simulated chirp mass posterior width. Can be 'none' (sets sigma_Mc = 0.01 Msun),
                               'dist' (scales with distance to merger), or 'snr' (scales with network signal-to-noise)
                               We use GW170817 as a reference point. (See Farr et al. (2016) for details on this scaling)
        Rbounds (tuple of floats) : (Rmin,Rmax); Min and max of R16 range. 
                                    If left unspecified, will be set to the min/max of a sampled prior, or (9km,15km) for a uniform prior.
        Ns (list of int) : List of N_events at which to plot the posterior evolution. 
                           If left unspecified, will be set to np.round(np.linspace(0,N_events,4))[1:].
        saveto (str) : '/path/to/save/directory/'
        showplots (bool) : Whether to display inline plots.
        nosignal (bool) : Whether to perform a signalless run for Monte Carlo validation.
        Rtrue (float) : True value of R_1.6. Used for plots. 
        bootstrap (array or str) : If None, empirical relation is assumed to be exact. If specified, must be an array of boostrapped 
                                   empirical relation coefficients. See empirical_relation_bootstrap() in hbpm_utils.py for details. 
                                   Can also be set to 'default', which loads the set of samples used for the Criswell+2022 paper.
        prior_bandwidth (float) : KDE bandwidth passed to Prior_f(). This is the scipy bw_method value.
        posterior_bandwidth (float) : KDE bandwidth passed to Posterior_f(). This is the scipy bw_method value.
        aggregation (str) : How to combine fpeak posteriors from different detectors. Can be 'sum' (adds samples together) or 'mult' (multiplies posterior kdes).
        ifos (str) : Which interferometers to use data from. Must be comma-delineated string. Can be any combination of H1, L1, and/or V1, e.g. 'H1,L1,V1'.
        seed (int) : Random seed to use for reproducability
    Returns:
        Rs (array) : Values of R_1.6 at which the likelihood and posterior are evaluated.
        likes (list of arrays) : List containing the individual event likelihood for all analyzed events.
        post (array) : Final multi-event posterior probability p(R|data) for each value of R in Rs.
        stats (tuple) : (mean,lower,upper) Mean and 95% confidence interval for R_1.6
        postdict (dict) : Posterior dictionary containing all information needed to reproduce the analysis.
    '''
    print("Running a hierarchical Bayesian post-merger analysis...")
    print("Loading priors...")
    if Rprior == 'uniform':
        if Rbounds is None:
            Rmin = 9
            Rmax = 15
        else:
            Rmin = Rbounds[0]
            Rmax = Rbounds[1]
        Rs = np.linspace(Rmin,Rmax,200)
        Rprior_kernel = st.uniform(loc=Rmin,scale=(Rmax-Rmin))
        R16prior = 'This analysis used a uniform prior on [{:2.1f} km,{:2.1f} km] for R_1.6.'.format(Rmin,Rmax)
    else:
        ## load Multimessenger prior
        R16prior = np.loadtxt(Rprior)
        if Rbounds is None:
            Rs = np.linspace(R16prior.min(),R16prior.max(),200)
        else:
            Rmin = Rbounds[0]
            Rmax = Rbounds[1]
            Rs = np.linspace(Rmin,Rmax,200)
        ## R16 prior KDE
        Rprior_kernel = kde(R16prior)
    ## fine grid
    Ms = np.linspace(0.8,1.8,200)
    
    ## allow direct passing of dict, or pre-sort by ifo
    if type(fprior_spec) is not dict:
        fprior_dict = {}
        for ifo in ifos.split(','):
            fprior_dict[ifo] = fprior_spec
    else:
        fprior_dict = fprior_spec
    ## now handle each ifo   
    for ifo in ifos.split(','):
        ## make sure user gets a helpful error if they haven't provided all the ifos we need
        if ifo not in fprior_dict.keys():
            raise TypeError("User-specified fprior_spec dictionary does not include all desired ifos. (Missing {})".format(ifo))
        ## grab the specification since we'll be overwriting it
        spec_i = fprior_dict[ifo]
        ## handle all supported possibilities
        if type(spec_i) is str:
            ## first option: load pickled data
            if spec_i.endswith('.pickle'):
                with open(spec_i,'rb') as file:
                    fprior_loaded = dill.load(file)
                if type(fprior_loaded) is dict:
                    for ifo_j in ifos.split(','):
                        fprior_dict[ifo_j] = fprior_loaded[ifo_j]
                        if fprior_dict[ifo_j].bw_method != prior_bandwidth:
                            print("Loading pre-processed fpeak prior for {}. Warning: prior_bandwidth argument will be ignored!".format(ifo_j))
                            print("Pre-computed prior bandwidth is {}.".format(fprior_dict[ifo_j].bw_method))
                else:
                    fprior_dict['combined'] = fprior_loaded
                    for ifo_j in ifos.split(','):
                        fprior_dict[ifo_j] = fprior_loaded
                break # get out of the for loop since we've gotten everything we need
            ## second option: uniform prior
            elif spec_i == 'uniform':
                fprior_dict[ifo] = st.uniform(loc=1.5,scale=2.5)
            ## as a last resort, load samples and construct prior
            else:
                fprior_samples = np.loadtxt(spec_i)
                fprior_dict[ifo] = Prior_f(fprior_samples,boundary='Reflection',kde_bandwidth=prior_bandwidth)
        elif type(spec_i) is np.ndarray:
            ## handles the array case
            fprior_samples = spec_i
            fprior_dict[ifo] = Prior_f(fprior_samples,boundary='Reflection',kde_bandwidth=prior_bandwidth)
        else:
            ## this handles a user-specified distribution
            ## make sure we have a .pdf() function attached
            if not hasattr(spec_i,'pdf'):
                raise TypeError("Specified prior distribution does not have a callable .pdf() function!")
            fprior_dict[ifo] = spec_i
     # get the final joint Prior_f object across all detectors
    if fprior_spec=='uniform':
        total_fprior = st.uniform(loc=1.5,scale=2.5)
    elif 'combined' in fprior_dict.keys():
        total_fprior = fprior_dict['combined']
    else:
        ## ensure prior arrays are all same size
        agg_prior_samples = [fprior_dict[ifo].samples for ifo in ifos.split(',')]
        lengths = [len(sample) for sample in agg_prior_samples]
        if np.all(lengths==lengths[0]):
            new_prior_samples = agg_prior_samples
        else:
            rng = np.random.default_rng(seed)
            ii, count = np.argmin(lengths), np.min(lengths)
            new_prior_samples = []
            for jj, samples in enumerate(agg_prior_samples):
                if jj==ii:
                    new_prior_samples.append(samples)
                else:
                    drop = len(samples) - count
                    new_prior_samples.append(np.delete(samples,rng.choice(len(samples),drop,replace=False)))
        for i, samples_i in enumerate(new_prior_samples):
            if i==0:
                agg_prior_samples = samples_i.reshape(-1,1)
            else:
                agg_prior_samples = np.hstack((agg_prior_samples, samples_i.reshape(-1,1)))
        if aggregation=='mult':
            final_samples = agg_prior_samples
        else:
            final_samples = agg_prior_samples.flatten()
        total_fprior = Prior_f(final_samples,boundary='Reflection',kde_bandwidth=prior_bandwidth)
    ## set Mchirp prior (uniform unless we change it)
    Mprior = st.uniform(loc=Ms.min(),scale=(Ms.max()-Ms.min()))
    
    ## get sampled empirical relations
    ## if bootstrap is None or an array, this is already handled, but this lets us specify 'default' or '/path/to/file.txt'.
    if type(bootstrap) is str:
        print("Loading sampled empirical relation coefficients...")
        bootstrap = load_bootstrap(bootstrap)
            
    
    print("Loading simulation data...")
    ## load external CSVs
    sim_df = pd.read_csv(sim_df_file)
    ev_df = pd.read_csv(ev_df_file)
    
    ## get Rtrue
    if Rtrue is None and eos is None:
        print('No Equation of State specified. Defaulting to SFHX EoS with Rtrue = 11.98 km.')
        eos = 'sfhx'
        Rtrue = 11.98
    elif Rtrue is not None and eos is not None:
        print('Warning: Both Equation of State and Rtrue have been independently specified and may or may not be compatible.')
    elif Rtrue is None and eos is not None:
        eos_dict = {'sfhx':11.98,'dd2':13.26,'sly4':11.54}
        if eos not in eos_dict.keys():
            print('Error: specified equation of state is unknown and no Rtrue has been provided.')
            raise
        Rtrue = eos_dict[eos]
    ## get event list
    if eos=='sly4':
        events = glob(datadir+'/'+'sly'+'_*')
    else:
        events = glob(datadir+'/'+eos+'_*')
    
    ## generate savenames
    if saveto is not None:
        ## make save directory
        os.mkdir(saveto)
        fpeak_subset_saveto = saveto+'/fpeak_recovery_subset.png'
        posterior_saveto = saveto+'/R16_posterior.png'
        evo_saveto = saveto+'/R16_posterior_evolution.png'
        postdict_saveto = saveto+'/posterior_eventdict.pickle'
    else:
        fpeak_subset_saveto = saveto
        posterior_saveto = saveto
        evo_saveto = saveto
        postdict_saveto = saveto
    
    ## handle ifos
    if type(ifos) is str:
        ifos = ifos.split(',')
    
    ## load data
    print("Generating event dictionary...")
    ## note that having mismatched prior types for different ifos isn't supported
    if fprior_spec == 'uniform':
        use_prior = fprior_spec
    elif np.any([not hasattr(fprior_dict[ifo],'rvs') for ifo in ifos]):
        print("Warning: user-specified prior distribution. No samples have been provided, so posterior draw-up will not be performed.")
        use_prior = 'no'
    else:
        use_prior = 'array'
    
    eventdict= gen_BayesWave_eventdict(events,ev_df,kde_boundary='Reflection',seed=seed,plot='subset',use_prior=use_prior,
                                           prior=fprior_dict,obs_run=True,sim_df=sim_df,nosignal=nosignal,
                                           saveto=fpeak_subset_saveto,showplot=showplots,
                                           title='Subset of BayesWave $\mathrm{f_{peak}}$ Posteriors',
                                           kde_bandwidth=posterior_bandwidth,z_adj=z_adj,Mchirp_scaling=Mchirp_scaling,
                                           aggregation=aggregation,ifos=ifos)

    ## compute likelihoods
    print("Computing likelihoods...")
    likes = get_multievent_likelihoods(Rs,Ms,eventdict,fprior=total_fprior,Mprior=Mprior,
                                       verbose=False,bootstrap=bootstrap,z_adj=z_adj,Mchirp_scaling=Mchirp_scaling)
    ## save
    if saveto is not None:
        print("Saving outputs to {}".format(saveto))
        postdict = save_posterior_eventdict(events,eventdict,ev_df,likes,R16prior,Rs,postdict_saveto,'BayesWave',returndict=True)
    
    print("Creating plots...")
    ## plot posterior
    plot_aggregate_posterior(Rs,likes,Rprior_kernel,Rtrue=Rtrue,glow=True,
                             saveto=posterior_saveto,showplot=showplots,
                             title='$\mathrm{R_{1.6}}$ Posterior')
    ## plot evolution with N_events
    if Ns is None:
        Ns = [int(i) for i in np.round(np.linspace(0,len(likes),4))[1:]]
    plot_posterior_by_Nevents(Rs,likes,Rprior_kernel,Ns,Rtrue=Rtrue,glow=False,
                              saveto=evo_saveto,showplot=showplots,
                              title='$\mathrm{R_{1.6}}$ Posterior by $\mathrm{N_{det}}$')
    ## get stats
    post = get_posterior(Rs,get_aggregate_likelihood(likes),Rprior_kernel)
    stats = get_post_stats(post,Rs)
    
    print("Done!")
    return Rs, likes, post, stats, postdict



if __name__ == '__main__':
    
    ## set up argparser
    parser = argparse.ArgumentParser(description='Run the hierarchical Bayesian post-merger analysis.')
    parser.add_argument('datadir', type=str, help='/path/to/BayesWave/Fpeak/output/head/directory/')
    parser.add_argument('sim_df_file', type=str, help='/path/to/csv/with/observing/run/event/details.csv')
    parser.add_argument('saveto', type=str, help='/path/to/save/directory/')
    parser.add_argument('--eos', type=str, 
                        help='Equation of state as used in BayesWave outputs. Must also specifiy --ev_df_file; the two should match. (default: sfhx)',
                        default='sfhx')
    parser.add_argument('--ev_df_file', type=str,
                        help='/path/to/csv/with/NR/simulation/details.csv Must also specifiy --eos; the two should match.(default: SFHX EoS datafile)',
                        default='./nr_files/sfhx_event_parameters.csv')
    parser.add_argument('--z_adj',stype=str,
                        help="If 'known', uses redshifts from ev_df_file to correct for redshifts assuming standard LambdaCDM.",
                        default=None)
    parser.add_argument('--Mchirp_scaling',stype=str,
                        help="Can be 'none' (all sigma_Mc=0.01Mo), 'dist' (distance scaling), or 'snr' (network SNR scaling).",
                        default='none')
    parser.add_argument('--fprior_file', type=str,
                        help='/path/to/file/with/fpeak/prior/samples.txt (default: BayesWave fpeak prior)',
                        default='./priors/fpeak_broad_prior.txt')
    parser.add_argument('--Rprior', type=str,
                        help="'/path/to/file/with/R16/prior/samples.txt' (default: Dietrich+2020 Multimessenger Prior) OR 'uniform'",
                        default='./priors/R16_prior.txt')
    parser.add_argument('-ns','--nosignal', help='Do an equivalent run without real signals for comparison.',
                        action='store_true')
    parser.add_argument('-plt','--showplots', help='Display inline plots.', action='store_true')
    parser.add_argument('--Rbounds', type=tuple,
                        help='(Rmin,Rmax) (default: Rprior sample min/max, or (9,15) for uniform Rprior)',
                        default=None)
    parser.add_argument('--N_thresholds', type=int, nargs='+',
                        help='N_events at which to plot R_1.6 posterior evolution. (defaults to 3 equally spaced intervals)',
                        default=None)
    parser.add_argument('--bootstrap', type=str,
                        help='/path/to/file/with/bootstrapped/empirical/relation/coefficients.txt (For incorporating relation error.)',
                        default=None)
    parser.add_argument('--prior_bandwidth', type=float,
                        help='KDE bandwidth for peak frequency prior. This is the scipy bw_method value.',
                        default=0.15)
    parser.add_argument('--post_bandwidth', type=float,
                        help='KDE bandwidth for peak frequency posterior. This is the scipy bw_method value.',
                        default=0.15)
    parser.add_argument('--aggregation', type=str,
                        help="How to aggregate fpeak data across ifos. Can be 'sum' (adds samples together) or 'mult' (multiplies posterior kdes)",
                        default='sum')
    parser.add_argument('--ifos', type=str,
                        help="Comma-delineated list of interferometers for fpeak data.",
                        default="H1,L1")
    args = parser.parse_args()
    
    ## deal with specific use cases
    if bool(args.eos) ^ bool(args.ev_df_file):
        parser.error('--eos and --ev_df_file must be given together. Default is SFHX.')
    if args.nosignal==True:
        print("Doing a no-signal run. Results will be based on simulated fpeak posteriors drawn from the fpeak prior.")
    if args.N_thresholds is None:
        Ns = args.N_thresholds
    else:
        Ns = list(args.N_thresholds)
    ## load bootstrap coefficients if needed
    if args.bootstrap is not None:
        print('Loading bootstrapped empirical relation coefficients...')
        ## should be mostly robust to different delimiters, but if you're having trouble switch to space-delimited
        boot_data = load_bootstrap(args.bootstrap)
    ## call function
    run_analysis(args.datadir,args.fprior_file,args.Rprior_file,args.eos,args.sim_df_file,args.ev_df_file,
                 Ns=Ns,saveto=args.saveto,showplots=args.showplots,bootstrap=boot_data,z_adj=args.z_adj,Mchirp_scaling=args.Mchirp_scaling,
                 prior_bandwidth=args.prior_bandwidth,post_bandwidth=args.post_bandwidth,aggregation=args.aggregation,ifos=args.ifos)
    
    
    
    
    