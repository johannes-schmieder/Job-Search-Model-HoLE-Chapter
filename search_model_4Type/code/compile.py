import sys
import os
import numpy as np # import numpy library
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
import time

import estimagic as em

from solvemodel import solveSingleTypeModel ,solveMultiTypeModel, matchingMoments, gmm
import output_latex
from output_latex import writeln


def compileModel(filename, estfile, estname):
    '''
    Generates a PDF document with simulation results.
        Arguments:
            logdir (str): Log directory
        Result:
            PDF document.
    '''
    #  ==== Define Colors ====
    blue    = tuple(np.array([9, 20, 145]) / 256)
    purple  = tuple(np.array([92,  6, 89]) / 256)
    fuchsia = tuple(np.array([155,  0, 155]) / 256)
    green   = tuple(np.array([0, 135, 14]) / 256)
    red     = tuple(np.array([128, 0, 2 ]) / 256)
    gray    = tuple(np.array([60, 60, 60 ]) / 256)

    target, cov, target_h12, target_h18, target_w12, target_w18 = matchingMoments()

    # === Log and figures directory ===

    T = 31
    P1 = 12
    P2 = 18
    ben1 = np.ones(T) * 800 / 30
    ben2 = np.ones(T) * 800 / 30
    ben1[:P1] = 1100 / 30
    ben2[:P2] = 1100 / 30
    inst1 = T, ben1
    inst2 = T, ben2
    timevec = np.arange(T) + 1

    logdir = '../log/'
    figpath = 'figures_'+filename+'/'
    logfile = filename + '.tex' # here it defines to which file the log will be written.

    if not os.path.isdir(logdir): # it checks if the log directory specified before exists or not. If not, it creates it.
        os.makedirs(logdir)
    if not os.path.isdir(logdir+figpath): # it checks if the log directory specified before exists or not. If not, it creates it.
        os.makedirs(logdir+figpath)

    # === Create Latex File ===
    pd.options.display.float_format = '{:,.5f}'.format
    # Latex file to log results

    output_latex.latex_header(logdir,logfile,'Search Model Simulations ')

    params_df = pd.read_excel(logdir + estfile, sheet_name='Sheet1', index_col=0)

    form =  '{:.4f}'
    output_latex.writeln(logdir,logfile,params_df.style.format_index(escape="latex").format(form).to_latex(
            caption='Parameters',
            position='h',
            position_float='centering',
            hrules=True
    ))

    # --- Store parameter vector in numpy array: ---
    params = params_df["value"].to_numpy()
    params = np.ndarray.flatten(params)

    h1,w1,S1,D1,W1 = solveMultiTypeModel(params,inst1)
    h2,w2,S2,D2,W2 = solveMultiTypeModel(params,inst2)

    # Return Moments
    dDdP = (D2-D1)/6 
    dWdP = (W2-W1)/6

    df_estopt = pd.read_excel(logdir+estname+'_estimation_options.xlsx'
                              , sheet_name='Sheet1', index_col=0)
    print(df_estopt)
    df_estopt.loc["dDdP"] = dDdP
    df_estopt.loc["dWdP"] = dWdP    

    print(df_estopt)
    # form =  '{:.4f}'
    output_latex.writeln(logdir,logfile,df_estopt.style.format_index(escape="latex").format(escape="latex").to_latex(
            caption='Estimation Options',
            position='h',
            position_float='centering',
            hrules=True
    ))

    df_start = pd.read_excel(logdir+estname+'_start_params.xlsx', sheet_name='Sheet1', index_col=0)
    df_start = df_start.rename(columns={'lower_bound': 'lower', 'upper_bound': 'upper'})
    form =  '{:.4f}'
    output_latex.writeln(logdir,logfile,df_start.style.format_index(escape="latex").format(form, escape="latex").to_latex(
            caption='Start Parameters',
            position='h',
            position_float='centering',
            hrules=True
    ))

    logdb = logdir  + estname + '_log.db'
    print(logdb)
    if os.path.exists(logdb):
        print('Found Logging Database \n')
        fig = em.criterion_plot(
            logdb,
            stack_multistart=False,
            monotone=True, #Whether to combine multistart histories into a single history. Default is False.
            show_exploration=False #If True, exploration samples of a multistart optimization are visualized. Default is False.
        )

        print(logdir+figpath+"criterion_plot.pdf")
        fig.write_image(logdir+figpath+"criterion_plot.pdf")

        with open(logdir+logfile,'a') as f:
            f.write('\\begin{figure}[h] \n')
            f.write('\\centering \n')
            f.write('\\caption{Criterion Plot } \n')
            f.write('\\includegraphics[clip=true, trim=.5cm .7cm .5cm 2.5cm,width =.7\\textwidth]{'+figpath+'criterion_plot.pdf} \n')
            f.write('\\end{figure} \n')
            f.write('\\newpage \n')

    plt.clf()
    plt.axvline(x=12, color="grey", linestyle="dashed")
    plt.axvline(x=18, color="grey", linestyle="dashed")
    plt.plot(timevec, h1, label="Simulated Hazard, P=12", linestyle="dashed", color=blue)
    plt.plot(timevec, h2, label="Simulated Hazard, P=18", linestyle="-.", color=red)
    plt.plot(timevec[:30], target_h12, label="Moments Hazard, P=12", linestyle="solid", color=blue, marker="o", markersize=4)
    plt.plot(timevec[:30], target_h18, label="Moments Hazard, P=18", linestyle="solid", color=red, marker="D", markersize=4)
    plt.ylabel("Exit Hazard")
    plt.xlabel("Months")
    plt.legend(loc="upper right")

    plt.ylim(bottom=0, top=0.15)
    # plt.show()

    title = 'Hazard Rate'
    figname = 'fig_haz.pdf'
    plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
    writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
    writeln(logdir,logfile, \
            '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.68\\textwidth]{' + figpath + figname + '} \n')

    plt.clf()
    plt.axvline(x=12, color="grey", linestyle="dashed")
    plt.axvline(x=18, color="grey", linestyle="dashed")
    plt.plot(timevec[:25], w1[:25], label="Simulated Wage, P=12", linestyle="dashed", color=blue)
    plt.plot(timevec[:25], w2[:25], label="Simulated Wage, P=18", linestyle="-.", color=red)
    plt.plot(timevec[:24], target_w12[:24], label="Moments Wage, P=12", linestyle="solid", color=blue, marker="o", markersize=4)
    plt.plot(timevec[:24], target_w18[:24], label="Moments Wage, P=18", linestyle="solid", color=red, marker="D", markersize=4)
    plt.ylabel("Log Wage")
    plt.xlabel("Months")
    plt.legend(loc="upper right")

    # plt.ylim(bottom=0, top=0.15)
    # plt.show()

    title = 'Reemployment Wage'
    figname = 'fig_wage.pdf'
    plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
    # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
    writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
    writeln(logdir,logfile, \
            '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.68\\textwidth]{' + figpath + figname + '} \n')

    # === Single Types ===
    delta, k, gamma, mu1, sigma, kappa, pi, k2, k3, k4, mu2, mu3, mu4, q2, q3, q4 = params

    # Parameters for single type model
    params1 = np.copy(params[0:7])

    # Show Type Aggregation for 2 types - HoLE Figure 1
    if 1:
        params1[1] = k
        params1[3] = mu1
        s1, logphi1, haz1, logw1, surv1, D1, Ew1 = solveSingleTypeModel(params1, inst1)
        params1[1] = k2
        params1[3] = mu2
        s2, logphi2, haz2, logw2, surv2, D2, Ew2 = solveSingleTypeModel(params1, inst1)
        params1[1] = k3
        params1[3] = mu3
        s3, logphi3, haz3, logw3, surv3, D3, Ew3 = solveSingleTypeModel(params1, inst1)
        params1[1] = k4
        params1[3] = mu4
        s4, logphi4, haz4, logw4, surv4, D4, Ew4 = solveSingleTypeModel(params1, inst1)        

        writeln(logdir,logfile,'\\pagebreak \n \\textbf{Show Type Aggregation} \n ')

        title = "Search Effort"
        plt.clf()
        plt.figure(figsize=(6, 3))  # Adjust the figure size here
        plt.plot(timevec, s1, label="Search effort, Type 1", linestyle="dashed", color=blue)
        plt.plot(timevec, s2, label="Search effort, Type 2", linestyle="-.", color=red)
        if q3>0:
            plt.plot(timevec, s3, label="Search effort, Type 3", linestyle="dotted", color=green)
        if q4>0:
            plt.plot(timevec, s4, label="Search effort, Type 4", linestyle="solid", color=purple)            
        # plt.title(title)
        plt.xlabel("Months")
        plt.ylabel(title)
        plt.legend(loc="center right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.45)
        figname = 'fig_type_agg_s.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} ')

        title = "Log Reservation Wage"
        plt.clf()
        plt.plot(timevec, logphi1, label="Log Res. Wage, Type 1", linestyle="dashed", color=blue)
        plt.plot(timevec, logphi2, label="Log Res. wage, Type 2", linestyle="-.", color=red)
        if q3>0:
                plt.plot(timevec, logphi3, label="Log Res. wage, Type 3", linestyle="dotted", color=green)
        if q4>0:
                plt.plot(timevec, logphi4, label="Log Res. wage, Type 4", linestyle="solid", color=purple)                
        # plt.title(title)
        plt.xlabel("Months")
        plt.ylabel(title)
        plt.legend(loc="center right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.10)
        figname = 'fig_type_agg_phi.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} \n')
        writeln(logdir,logfile,' \n\n ')

        title = "Survival Function"
        plt.clf()
        plt.plot(timevec, surv1, label="Survival function, Type 1", linestyle="dashed", color=blue)
        plt.plot(timevec, surv2, label="Survival function, Type 2", linestyle="-.", color=red)
        if q3>0:
                plt.plot(timevec, surv3, label="Survival function, Type 3", linestyle="dotted", color=green)
        if q4>0:
                plt.plot(timevec, surv4, label="Survival function, Type 4", linestyle="solid", color=purple)                
        # plt.title(title)
        plt.xlabel("Months")
        plt.ylabel(title)
        plt.legend(loc="upper right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.10)
        figname = 'fig_type_agg_surv.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} ')

        title = "Type Shares"
        plt.clf()
        q1 = 1 - q2 - q3 - q4
        weight1 = q1 * surv1 / (q1 * surv1 + q2 * surv2+ q3 * surv3+ q4 * surv4)
        weight2 = q2 * surv2 / (q1 * surv1 + q2 * surv2+ q3 * surv3+ q4 * surv4)
        weight3 = q3 * surv3 / (q1 * surv1 + q2 * surv2+ q3 * surv3+ q4 * surv4)
        weight4 = q4 * surv4 / (q1 * surv1 + q2 * surv2+ q3 * surv3+ q4 * surv4)

        plt.plot(timevec, weight1, label="Share among survivors, Type 1", linestyle="dashed", color=blue)
        plt.plot(timevec, weight2, label="Share among survivors, Type 2", linestyle="-.", color=red)
        if q3>0:
                plt.plot(timevec, weight3, label="Share among survivors, Type 3", linestyle="dotted", color=green)
        if q4>0:
                plt.plot(timevec, weight4, label="Share among survivors, Type 4", linestyle="solid", color=purple)
        # plt.title(title)
        plt.xlabel("Months")
        plt.ylabel(title)
        plt.legend(loc="center right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.10)
        figname = 'fig_type_agg_share.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} ')

        writeln(logdir,logfile,' \n\n ')

        title = "Exit Hazard"
        plt.clf()
        plt.plot(timevec, haz1, label="Hazard, Type 1", linestyle="dashed", color=blue)
        plt.plot(timevec, haz2, label="Hazard, Type 2", linestyle="-.", color=red)
        if q3>0: 
                plt.plot(timevec, haz3, label="Hazard, Type 3", linestyle="dotted", color=green)
        if q4>0: 
                plt.plot(timevec, haz4, label="Hazard, Type 4", linestyle="solid", color=purple)                
        # plt.title(title)
        plt.ylabel(title)
        plt.xlabel("Months")
        plt.legend(loc="center right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.10)
        figname = 'fig_type_agg_hazard.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} ')
        # writeln(logdir,logfile,' \n\n ')

        title = "Log Reemployment Wage "
        plt.clf()
        plt.plot(timevec, logw1, label="Log Reemp. Wage,  Type 1", linestyle="dashed", color=blue)
        plt.plot(timevec, logw2, label="Log Reemp. wage,  Type 2", linestyle="-.", color=red)
        if q3>0:
                plt.plot(timevec, logw3, label="Reemp. wage,  Type 3", linestyle="dotted", color=green)
        if q4>0:
                plt.plot(timevec, logw4, label="Reemp. wage,  Type 4", linestyle="solid", color=purple)                
        plt.ylabel(title)
        plt.xlabel("Months")
        plt.legend(loc="center right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.10)
        figname = 'fig_type_agg_w.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} \n')
        writeln(logdir,logfile,' \n\n ')

    # Effects of UI - HoLE Figure 1
    if 1: 
        h1,w1,S1,D1,W1 = solveMultiTypeModel(params,inst1)

        plt.clf()
        plt.plot(timevec, h1, label="Hazard", linestyle="dashed", color=blue)
        # plt.plot(timevec[:30], target_h12, label="Moments Hazard, P=12", linestyle="solid", color=blue)
        # plt.title("Aggregate Hazard")
        plt.xlabel("Months")
        plt.ylabel("Exit Hazard")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.12)    
        title = 'Hazard Rate'
        figname = 'fig_haz_agg.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '}')
        # writeln(logdir,logfile,' \n\n ')

        plt.clf()
        plt.plot(timevec[:25], w1[:25], label="Log Reemployment Wage", linestyle="dashed", color=blue)
        # plt.plot(timevec[:24], target_w12[:24], label="Moments Wage, P=12", linestyle="solid", color=blue)
        # plt.plot(timevec[:24], target_w18[:24], label="Moments Wage, P=18", linestyle="solid", color=red)
        # plt.title("Aggregate Reemployment Wage")
        plt.xlabel("Months")
        plt.ylabel("Log Wage")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        # plt.show()
        title = 'Reemployment Wage'
        figname = 'fig_wage_agg.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} \n')

    for type in [1,2,3]:
        if type == 1:
            params1[1] = k
            params1[3] = mu1
        if type == 2: 
            params1[1] = k2
            params1[3] = mu2
        if type == 3:
            params1[1] = k3
            params1[3] = mu3

        s1, logphi1, haz1, logw1, surv1, D1, Ew1 = solveSingleTypeModel(params1, inst1)
        s2, logphi2, haz2, logw2, surv2, D2, Ew2 = solveSingleTypeModel(params1, inst2)

        writeln(logdir,logfile,'\\pagebreak \n \\textbf{Type '+ str(type) + '} \n ')

        title = "Search Effort - Type "+str(type) 
        plt.clf()
        plt.figure(figsize=(6, 4.5))  # Adjust the figure size here
        plt.plot(timevec, s1, label="Search effort, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, s2, label="Search effort, P=18", linestyle="-.", color=red)
        # plt.title(title)
        plt.xlabel("Months")
        plt.ylabel("Search Effort")
        plt.legend(loc="lower right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="-.")
        # plt.ylim(bottom=0, top=0.10)
        figname = 'fig_type'+str(type)+'_s.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} ')

        title = "Reservation Wage - Type "+str(type) 
        plt.clf()
        plt.plot(timevec, logphi1, label="Log Res. Wage, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, logphi2, label="Log Res. wage, P=18", linestyle="-.", color=red)
        # plt.title(title)
        plt.xlabel("Months")
        plt.ylabel("Log Reservation Wage")
        plt.legend(loc="upper right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="-.")
        # plt.ylim(bottom=0, top=0.10)
        figname = 'fig_type'+str(type)+'_phi.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} \n')

        title = "Exit Hazard - Type "+str(type) 
        plt.clf()
        plt.plot(timevec, haz1, label="Hazard, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, haz2, label="Hazard, P=18", linestyle="-.", color=red)
        # plt.title(title)
        plt.xlabel("Months")
        plt.ylabel("Exit Hazard")
        plt.legend(loc="lower right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="-.")
        # plt.ylim(bottom=0, top=0.10)
        figname = 'fig_type'+str(type)+'_hazard.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} ')

        title = "Reemployment Wage - Type "+str(type) 
        plt.clf()
        plt.plot(timevec, logw1, label="Log Reemp. Wage, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, logw2, label="Log Reemp. wage, P=18", linestyle="-.", color=red)
        # plt.title(title)
        plt.xlabel("Months")
        plt.ylabel("Log Wage")
        plt.legend(loc="upper right")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="-.")
        # plt.ylim(bottom=0, top=0.10)
        figname = 'fig_type'+str(type)+'_w.pdf'
        plt.savefig(logdir+figpath+figname, bbox_inches="tight")      
        # plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        # writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.45\\textwidth]{' + figpath + figname + '} \n')

    # Close compiler
    output_latex.latex_close(logdir,logfile)


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=2)
    np.set_printoptions(linewidth=152)

    print('\n\n\n')
    # Set number of decimals when printing numpy arrays
    np.set_printoptions(linewidth=152)
    np.set_printoptions(precision=3, suppress= True, threshold=sys.maxsize)


    print('\n\n\n')

  
    estname = 'Est1'
    compileModel(
        filename= estname +'_compiled',
        estfile = estname +'.xlsx',
        estname = estname 
        )
