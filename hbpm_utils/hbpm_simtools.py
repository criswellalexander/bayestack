#!/usr/bin/env python
# coding: utf-8

'''
This file contains various functions for processing numerical relativity simulations for use with the Hierarchical Post-Merger Bayesian analysis (HBPM). 
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
import pathlib
from tabulate import tabulate
import pickle
sys.path.append(os.path.abspath('.'))
from hbpm_utils import *



'''
Table of Contents

1. Observing Run Simulations

'''

## some preliminaries; setting up default paths and the like

## paths to NR sim info (not sims themselves)
topdir = str(pathlib.Path(__file__).parents[1].absolute())
default_nr_paths = {
    'sfhx':os.path.abspath(topdir+'/nr_files/sfhx_event_parameters.csv'),
    'dd2':os.path.abspath(topdir+'/nr_files/dd2_event_parameters.csv'),
    'sly4':os.path.abspath(topdir+'/nr_files/sly4_event_parameters.csv')
}
## paths to observing run simulation info
default_obs_paths = {
    'O4':os.path.abspath(topdir+'/observing_run_sims/O4_events_updated.csv'),
    'O5':os.path.abspath(topdir+'/observing_run_sims/O5_events_updated.csv')
}


## SECTION 1: OBSERVING RUN SIMULATIONS

def obsrun_to_nr(eos,run,obs_path=None,nr_path=None,saveto=None,showplot=False,fixnames=False):
    '''
    Function to take a list of events for an observing run simulation and, for each event, find the closest NR simulation of 
    a given equation of state in both m_1 and m_2.
    
    Arguments:
        eos (str) : Equation of State name. Can be sfhx, dd2, or sly4.
        run (str) : Observing run. Can be 'O4' or 'O5'.
        obs_path (str) : '/path/to/obs_run_params.csv', a file containing the desired event parameters for each event. Needs at least
                         distance, mass1, mass2, mchirp, q, inclination, ifos, and snr. Assumes Petrov+2021 format. Defaults to files
                         listed in the default_obs_paths dictionary at top of hbpm_simtools.
        nr_path (str) : '/path/to/nr_sim_params.csv', a file containing the NR simulation parameters for the specified EoS. Defaults to
                        files listed in the default_nr_paths dictionary at top of hbpm_simtools.
        saveto (str) : '/path/to/save/file.csv', location to save final csv, if desired.
        showplot (bool) : Whether to plot the NR sim parameters vs. simulated event parameters.
        fixnames (bool) : Whether to fix the naming convention in the filenames for use with BayesWave
    Returns:
        events_df (pandas Dataframe) : Dataframe containing all relevant info for both the original event and the liaised NR sim.
    '''
    ## set paths if unspecified
    if obs_path is None:
        obs_path = default_obs_paths[run]
    if nr_path is None:
        nr_path = default_nr_paths[eos]
    
    ## load csvs
    obs_df = pd.read_csv(obs_path)
    nr_df = pd.read_csv(nr_path)
    
    ## get closest nr sim to observed event using a 2D Euclidian norm in m_1 and m_2
    corr = []
    mstr = []
    file_q = []
    for m1, m2, mchirp, q in zip(obs_df['mass1'],obs_df['mass2'],obs_df['mchirp'],obs_df['q']):
        ## for soultanis sims, m2 > m1 always; not so for O4/O5 parameters, so adjust the order
        m_small = np.minimum(m1,m2) ## Soultanis m1
        m_large = np.maximum(m1,m2) ## Soultanis m2
        dist = np.sqrt((nr_df['m1'] - m_small)**2 + (nr_df['m2'] - m_large)**2)
        bestfile = nr_df['file'][np.argmin(dist)]
        corr.append(bestfile)
        mstr.append(parse_datafile(bestfile))
        file_q.append(parse_qm1m2(bestfile)[0])
        
    if showplot==True:
        m_small = np.minimum(obs_df['mass1'],obs_df['mass2']) ## Soultanis m1
        m_large = np.maximum(obs_df['mass1'],obs_df['mass2']) ## Soultanis m2
        m1_corr = [nr_df[nr_df['file']==name]['m1'].to_numpy() for name in corr]
        m2_corr = [nr_df[nr_df['file']==name]['m2'].to_numpy() for name in corr]
        plt.figure()
        plt.scatter(m_small,m_large,label='Events')
        plt.scatter(m1_corr,m2_corr,label='NR Sims')
        plt.ylabel('$m_2$')
        plt.xlabel('$m_1$')
        plt.legend()
        plt.show()
        
    ## copy obs dataframe and add NR file
    events_df = obs_df.copy()
    if fixnames==True:
        corr_fixed = [name.replace('dd2-','dd2_').replace('sly-','sly_').replace('mttot','mtot') for name in corr]
        events_df['waveform'] = corr_fixed
    else:
        events_df['waveform'] = corr
    ## add mass string and reformat q
    events_df['mstr'] = mstr
    events_df['q_waveform'] = file_q
    events_df['q_waveform'] = events_df['q_waveform'].apply('{:0.2f}'.format)
    events_df['ifos'] = events_df['ifos'].str.replace(',','-')
    
    ## save to csv if desired
    if saveto is not None:
        events_df.to_csv(saveto,index=False)
    
    return events_df


def get_tables(events_df,table_type):
    '''
    Function to print latex tables for observing runs.
    
    Arguments:
        events_df (pandas Dataframe) : Dataframe as produced by obsrun_to_nr(), above.
        table_type (str) : Kind of table. Can be 'events' (shows observed event parameters) or 'sims' (shows liaised NR sim parameters)
    Returns:
        printed output: latex-ready table
    '''
    ## define columns
    ev_cols = ['simulation_id','mass1','mass2','q','mchirp','distance','snr']
    sim_cols = ['simulation_id','waveform']
    ## float formatting for the longer floats
    ## 0.0f is a placeholder for the (string) filename
    float_fmts_evs = ['0.0f','0.3f','0.3f','0.2f','0.3f','0.2f','0.1f']
    float_fmts_sims = ['0.0f','0.0f','0.3f','0.3f','0.2f','0.3f']
    if table_type=='events':
        paper_table_evs = events_df.copy()[ev_cols]
        print(tabulate(paper_table_evs,headers='keys',showindex='never',tablefmt='latex_longtable',floatfmt=float_fmts_evs))
    elif table_type=='sims':
        paper_table_sims = events_df.copy()[sim_cols]
        qs, m1s, m2s = parse_qm1m2(paper_table_sims['waveform'])
        paper_table_sims['waveform_m1'] = m1s
        paper_table_sims['waveform_m2'] = m2s
        paper_table_sims['waveform_q'] = qs
        paper_table_sims['waveform_Mchirp'] = calc_Mchirp(paper_table_sims['waveform_m1'],paper_table_sims['waveform_m2'])
        print(tabulate(paper_table_sims,headers='keys',showindex='never',tablefmt='latex_longtable',floatfmt=float_fmts_sims))
    
    return
        










