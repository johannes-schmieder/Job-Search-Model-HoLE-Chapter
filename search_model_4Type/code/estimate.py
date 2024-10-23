# -*- coding: utf-8 -*-
# Author: Johannes Schmieder
# EC 751 - Labor Economics - Problem Set 2 - Part 2

import sys
import os
import numpy as np # import numpy library
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
import time

import estimagic as em

from solvemodel import solveModel ,solveMultiTypeModel, matchingMoments, gmm

from compile import compileModel

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=152)

print('\n\n\n')
# Set number of decimals when printing numpy arrays
np.set_printoptions(linewidth=152)
np.set_printoptions(precision=3, suppress= True, threshold=sys.maxsize)

#  ==== Define Colors ====
blue    = tuple(np.array([9, 20, 145]) / 256)
purple  = tuple(np.array([92,  6, 89]) / 256)
fuchsia = tuple(np.array([155,  0, 155]) / 256)
green   = tuple(np.array([0, 135, 14]) / 256)
red     = tuple(np.array([128, 0, 2 ]) / 256)
gray    = tuple(np.array([60, 60, 60 ]) / 256)


if __name__ == "__main__":

    logdir = './log/'
    if not os.path.isdir(logdir): # it checks if the log directory specified before exists or not. If not, it creates it.
        os.makedirs(logdir)

    print("\n\n\n")
    # Set number of decimals when printing numpy arrays
    np.set_printoptions(linewidth=152)
    np.set_printoptions(precision=2, suppress=True, threshold=sys.maxsize)

    #  ==== Define Colors ====
    blue = tuple(np.array([9, 20, 145]) / 256)
    purple = tuple(np.array([92, 6, 89]) / 256)
    fuchsia = tuple(np.array([155, 0, 155]) / 256)
    green = tuple(np.array([0, 135, 14]) / 256)
    red = tuple(np.array([128, 0, 2]) / 256)
    gray = tuple(np.array([60, 60, 60]) / 256)

    algo = "scipy_lbfgsb"

    T = 31
    P1 = 12
    P2 = 18
    ben1 = np.ones(T) * 800 / 30
    ben2 = np.ones(T) * 800 / 30
    ben1[:P1] = 1100 / 30
    ben2[:P2] = 1100 / 30
    inst1 = T, ben1
    inst2 = T, ben2
    timevec = np.arange(T)

    target, cov, target_h12, target_h18, target_w12, target_w18 = matchingMoments()
    W = np.linalg.inv(cov)
    print(W[:5,:5])

    if 1:
        delta = 0.995 
        gamma = 0.545 
        mu1 = 3.4
        sigma = 0.25
        kappa = 24 
        pi = 0
        k1 = 700
        k2 = 10
        k3 = 1
        k4 = 1
        mu2 = 3.4 
        mu3 = 2.5
        mu4 = 3
        q2 = .5
        q3 = 0
        q4 = 0
        params_full = pd.DataFrame(
            data={
                "value": [delta, k1, gamma, mu1, sigma, kappa, pi,k2,k3,k4,mu2,mu3,mu4,q2,q3,q4]
            },
            index=["delta", "k1", "gamma", "mu1", "sigma", "kappa", "pi", "k2", "k3","k4", "mu2", "mu3","mu4", "q2", "q3", "q4"]
    ,
        )
        gmm_object = gmm(params_full.copy(),target, W, inst1, inst2, disp=True)

        params = pd.DataFrame(
            data={
            "value"       : [0.8], "lower_bound" : [0.1],  "upper_bound" : [0.999]
            },
            index=["delta"],
        )

        # print(gmm_object.sse(params))
        # params = pd.DataFrame(
        #     data={
        #         "value":       [100, 200, 50, 50  ,3, 3.6, 3.4, 3, 0.2, 0.2, 0.2, 0.2, 0.1,0.995],
        #         "lower_bound": [0.1, 0.1, .1, .1  ,2, 2, 2, 2, 0, 0, 0, 0.01, 0.01, 0.8],
        #         "upper_bound": [200, 200, 200, 200,5, 5, 5, 5, .33, .33, .33, 1.0, 1, 0.999],
        #     },
        #     index=["k1", "k2", "k3","k4", "mu1", "mu2", "mu3","mu4", "q2", "q3","q4", "gamma", "sigma","delta"],
        # )
        # print(gmm_object.sse(params))
        # print(gmm_object.criterion(params))
        # gmm_object = gmm(params_full.copy(), target, W, inst1, inst2, disp=True)
        # gmm_object.params_full.update(params)
        

    if 1:
        print('========== SSE ==========')
        print(gmm_object.criterion(params))
        print('sse')
        print(gmm_object.sse(params))
        print('=============')

        # const={
        #         "loc": [6,7],
        #         "type": "linear",
        #         "lower_bound": 0.,
        #         "upper_bound": 1.,
        #         "weights": 0.5,
        #     }
        # const={"loc": [6,7], "type": "probability"}
        # print(gmm_object.sse(params))
        # print(gmm_object.criterion(params))
        # gmm_object = gmm(params_full.copy(),target, W, inst1, inst2, disp=True)
        # gmm_object.params_full.update(params)
        # print(solveMultiTypeModel(gmm_object.params_full["value"].to_numpy(),inst2))
        # estname = 'Est4'

        estname = 'Est1'
        multistart = True
        multistart = False
        # algo = "tao_pounders"
        # algo = "scipy_lbfgsb"
        algo = "nag_dfols"
        # algo = "pounders"
        N_cpu = 28
        samples_per_core = 500
        n_samples = N_cpu * samples_per_core

        print('Save Optimization Options')
        estimation_settings = {
            "spec": [estname],
            "algo": [algo],
            "multistart": [multistart],
            "n_cores": [N_cpu],
            "n_samples" : [n_samples],
            "Finished" : [False]
        }

        df_estopt = pd.DataFrame.from_dict(
                estimation_settings,
                orient='index',
                columns=['Option',])

        df_estopt.to_excel(logdir+estname+'_estimation_options.xlsx')

        estimation_settings['Finished'] = True
        tic=time.time()
        if multistart==False:
            algo = "scipy_lbfgsb"
            print('\n === Estimation using Estimagic === \n')
            print('Algorithm: ' + algo + '\n')
            print('Multistart: False \n')
            res = em.minimize(
                criterion=gmm_object.criterion,
                # criterion=gmm_object.sse,
                params=params,
                scaling=True,
                scaling_options={"method": "bounds", "clipping_value": 0.0},
                algorithm=algo,
                logging=logdir + estname + "_log.db",
                log_options={
                    "fast_logging": True,
                    "if_database_exists": "replace",  # one of "raise", "replace", "extend",
                    "if_table_exists": "replace",  # one of "raise", "replace", "extend"
                },
            )
        if multistart==True:
            print('\n === Estimation using Estimagic === \n')
            print('Algorithm: ' + algo + '\n')
            print('Multistart: True \n')
            res = em.minimize(
                criterion=gmm_object.criterion,
                params=params,
                scaling=True,
                scaling_options={"method": "bounds", "clipping_value": 0.0},
                algorithm=algo,
                multistart=True,
                multistart_options = {
                    "n_cores"  : N_cpu,
                    "n_samples": n_samples,
                    "sampling_method" : 'latin_hypercube',
                    "convergence_max_discoveries" : 2,
                    "share_optimizations" : .1
                },
                logging=logdir + estname + "_log.db",
                log_options={
                    "fast_logging": True,
                    "if_database_exists": "replace",  # one of "raise", "replace", "extend",
                    "if_table_exists": "replace",  # one of "raise", "replace", "extend"
                },
            )

        toc = time.time()-tic

        print('Estimated Parameters:')
        print(res.params)
        print(res.criterion)
        print(res)
        col0 = res.params['value']
        col0.loc['SSE'] = res.criterion
        em.criterion_plot(res,monotone=True)

        print(gmm_object.params_full)
        gmm_object.params_full.update(res.params)
        df = gmm_object.params_full
        df.to_excel(logdir + estname +'.xlsx')

        df_estopt.loc['Finished']=True
        # df_estopt.loc['Exception']=exception_occured
        # df_estopt.loc['Exception msg']=exc
        df_estopt.loc['Runtime (hours)']=toc/3600
        df_estopt.loc['Criterion']=res.criterion
        df_estopt.loc['Criterion Evaluations']=res.n_criterion_evaluations

        print(df_estopt)
        df_estopt.to_excel(logdir+estname+'_estimation_options.xlsx')
        res.start_params.to_excel(logdir+estname+'_start_params.xlsx')

        compileModel(
            filename= estname +'_compiled',
            estfile = estname +'.xlsx',
            estname = estname 
            )
