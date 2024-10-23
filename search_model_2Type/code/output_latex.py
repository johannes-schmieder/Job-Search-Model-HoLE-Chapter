#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool to create tex file with results

@author: johannes
"""

import os

#fid = fopen(fullfile(logdir,[outputfile '.tex']),'w');
#
#output_latex_header(fid,title,h_0,1,w,c_0,1,1,1,1,nu,0)

def latex_header(logdir,logfile,title):
    
    try:
        os.remove(logdir+logfile)
    except: 
        print('Latex log file either does not exist or could not be removed')
    
    with open(logdir+logfile,'a') as f:
        f.write('\\documentclass[letter, 10pt]{article}\n')
        f.write('\\usepackage{amssymb,amsmath,amsthm,verbatim,esint,multicol}\n')
        f.write('\\usepackage{enumitem}\n')
        f.write('\\usepackage[round]{natbib}\n')
        f.write('\\usepackage[margin=.66 in]{geometry}\n')
        f.write('\\usepackage{fancyhdr}\n')
        f.write('\\usepackage{titlesec}\n')
        f.write('\\usepackage{bbm}\n')
        f.write('\\usepackage{graphicx}\n')
        f.write('\\usepackage{subcaption}\n')
        f.write('\\usepackage{pdflscape}\n')
        f.write('\\usepackage{color}\n')
        f.write('\\usepackage{amsmath}\n')
        f.write('\\usepackage{footnote}\n')
        f.write('\\makesavenoteenv{tabular}\n')
        f.write('\\usepackage{booktabs,threeparttable}\n')
        f.write('\\usepackage{setspace}\n')
        f.write('\\titleformat*{\\section}{\\large\\bfseries}\n')
        f.write('\\titleformat*{\\subsection}{\\bfseries}\n')
        f.write('\\titleformat*{\\subsubsection}{\\itshape}\n')
        f.write('\\pagestyle{fancy}\n')
        f.write('\\fancyhead{}\n')
        f.write('\\fancyfoot{}\n')
        f.write('\\fancyhead[CO,CE]{'+title+'}\n')
        f.write('\\setlength\\parindent{24pt}\n')
        f.write('\\begin{document}\n')
        # f.write('\\onehalfspacing\n')
        # f.write('\\section{'+title+'}\n')
   
def latex_close(logdir,logfile):
    
    with open(logdir+logfile,'a') as f:        
        f.write('\\end{document}\n')

    origdir = os.getcwd()
    os.chdir(logdir)
    
    if os.name=='posix' :
        if os.path.exists('/projectnb/welfgr/'):
            print('On Cluster!')
            os.system('/share/pkg.7/texlive/2018/install/bin/x86_64-linux/pdflatex -shell-escape --src -interaction=nonstopmode '+logfile)
        else: 
            print('On my Mac')
            os.system('/Library/TeX/texbin/pdflatex -shell-escape --src -interaction=nonstopmode '+logfile)
    elif os.name=='nt' :
        os.system('pdflatex -shell-escape --src -interaction=nonstopmode '+logfile)

    # Delete auxiliary files
    aux_files = [logfile[:-4] + ext for ext in ['.aux', '.log', '.out', '.toc', '.fdb_latexmk']]
    for aux_file in aux_files:
        try:
            os.remove(aux_file)
        except FileNotFoundError:
            print(f'{aux_file} not found, skipping deletion.')

    os.chdir(origdir)

    if os.name=='posix' :
        # s = 'open '+logdir+logfile[:-4]+'.pdf'
        os.system('open '+logdir+logfile[:-4]+'.pdf')
    elif os.name=='nt' :
        os.system('start '+logdir+logfile[:-4]+'.pdf')

def section(logdir,logfile,section_title):
    
    with open(logdir+logfile,'a') as f:
        f.write('\n \\section{' + section_title +'}')


        
def parameter_table(logdir,logfile,param_time,param_inst,param_pref):
    
    
    # Unpack parameters
    # Unpack parameters
    T_0,T_last,T_ERA,T_SRA,steps,maxP_mon = param_time
    p_H_mon,w_ret_mon,rho,kink,y_O_mon,y_U_mon,reprate,p,sigma = param_inst
    a,e,k,gamma,lam,kappa,beta_yearly,delta_yearly  = param_pref
    
    with open(logdir+logfile,'a') as f:
        f.write('\\begin{table}[h] \centering')
        f.write('\\caption{Parameters} \n')
        f.write('\\begin{tabular}{l *{2}{c}} \\\\ \n')
        f.write('\\toprule \n \n')
        f.write('Parameter & Value \\\\')
        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Time Parameters} \\\\ \n')
        
        f.write('T\\_0 & ' + '{:.2f}'.format(T_0) + ' \\\\ \n')
        f.write('T\\_last & ' + '{:.2f}'.format(T_last) + ' \\\\ \n')
        f.write('T\\_ERA & ' + '{:.2f}'.format(T_ERA) + ' \\\\ \n')
        f.write('T\\_SRA & ' + '{:.2f}'.format(T_SRA) + ' \\\\ \n')
        f.write('steps & ' + '{:.2f}'.format(steps) + ' \\\\ \n')
        f.write('maxP\\_mon & ' + '{:.2f}'.format(maxP_mon) + ' \\\\ \n')
        
        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Institutional Parameters} \\\\ \n')
        f.write('p\\_H\\_mon & ' + '{:.2f}'.format(p_H_mon) + ' \\\\ \n')
        f.write('w\\_ret\\_mon & ' + '{:.2f}'.format(w_ret_mon) + ' \\\\ \n')
        f.write('rho & ' + '{:.2f}'.format(rho) + ' \\\\ \n')
        f.write('kink & ' + '{:.2f}'.format(kink) + ' \\\\ \n')
        f.write('y\\_O\\_mon & ' + '{:.2f}'.format(y_O_mon) + ' \\\\ \n')
        f.write('y\\_U\\_mon & ' + '{:.2f}'.format(y_U_mon) + ' \\\\ \n')
        f.write('reprate & ' + '{:.2f}'.format(reprate) + ' \\\\ \n')
        
        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Preference Parameters} \\\\ \n')
        f.write('a & ' + '{:.2f}'.format(a) + ' \\\\ \n')
        f.write('e & ' + '{:.2f}'.format(e) + ' \\\\ \n')
        f.write('k & ' + '{:.2f}'.format(k) + ' \\\\ \n')
        f.write('gamma & ' + '{:.2f}'.format(gamma) + ' \\\\ \n')
        f.write('lam & ' + '{:.2f}'.format(lam) + ' \\\\ \n')
        f.write('kappa & ' + '{:.2f}'.format(kappa) + ' \\\\ \n')
        f.write('beta\\_yearly & ' + '{:.2f}'.format(beta_yearly) + ' \\\\ \n')
        f.write('delta\\_yearly & ' + '{:.2f}'.format(delta_yearly) + ' \\\\ \n')

        f.write('\\bottomrule \n \n')
        f.write('\\end{tabular} \n')
        f.write('\\end{table} \n')
    
        # Time Parameters / all integers
    T_0    = 50    # Age of first period
    T_last = 70    # Age of last period, total periods T_last-T_0+1
    T_ERA  = 65    # Time at which early retirement via UI is possible
    T_SRA  = 68    # Standard retirement age
    steps  = 2     # Number of steps into which each age year is split up
    maxP   = int(1*steps)  # Maximum Potential Benefit Duration

    # Institutional Parameters
    p_H    = 1000.  # high productivity
    w_ret  = 1000.  # wage used for pension calculation
    rho    = 0.02   # Accrual rate of retirement benefits
    kink   = 0.5    # Kink in bundget set
    y_h    = 100.   # Home production when unemployed
    b      = 0.6 * w_ret # UI benefit level
    
    # Preference Parameters
    a = 1           # Level Parameter, disutility of work
    e = .6          # Elasticity of lifetime labor supply
    k = 2000000     # Cost level of job search cost
    gamma =4        # Curvature of search cost
    lam = 0.5       # Low productivity shock probability
    kappa = 0.5     # Bargaining weight worker
    beta = 0.995    # Discount factor workers
    delta = 0.995   # Discount factor firms
    
def parameter_tablePhi(logdir,logfile,param_time,param_inst,param_pref):
    
    
    # Unpack parameters
    # Unpack parameters
    T_0,T_last,T_ERA,T_SRA,steps,maxP_mon = param_time
    p_H_mon,w_ret_mon,rho,kink,y_O_mon,y_U_mon,reprate,p,sigma = param_inst
    a0,a1,a2,a3,k,gamma,lam,kappa,beta_yearly,delta_yearly  = param_pref
    
    with open(logdir+logfile,'a') as f:
        f.write('\\centering \n \\begin{table}[h]')
        f.write('\\caption{Parameters} \n')
        f.write('\\begin{tabular}{l *{2}{c}} \\\\ \n')
        f.write('\\toprule \n \n')
        f.write('Parameter & Value \\\\')
        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Time Parameters} \\\\ \n')
        
        f.write('T\\_0 & ' + '{:.2f}'.format(T_0) + ' \\\\ \n')
        f.write('T\\_last & ' + '{:.2f}'.format(T_last) + ' \\\\ \n')
        f.write('T\\_ERA & ' + '{:.2f}'.format(T_ERA) + ' \\\\ \n')
        f.write('T\\_SRA & ' + '{:.2f}'.format(T_SRA) + ' \\\\ \n')
        f.write('steps & ' + '{:.2f}'.format(steps) + ' \\\\ \n')
        f.write('maxP\\_mon & ' + '{:.2f}'.format(maxP_mon) + ' \\\\ \n')
        
        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Institutional Parameters} \\\\ \n')
        f.write('p\\_H\\_mon & ' + '{:.2f}'.format(p_H_mon) + ' \\\\ \n')
        f.write('w\\_ret\\_mon & ' + '{:.2f}'.format(w_ret_mon) + ' \\\\ \n')
        f.write('rho & ' + '{:.2f}'.format(rho) + ' \\\\ \n')
        f.write('kink & ' + '{:.2f}'.format(kink) + ' \\\\ \n')
        f.write('y\\_O\\_mon & ' + '{:.2f}'.format(y_O_mon) + ' \\\\ \n')
        f.write('y\\_U\\_mon & ' + '{:.2f}'.format(y_U_mon) + ' \\\\ \n')
        f.write('reprate & ' + '{:.2f}'.format(reprate) + ' \\\\ \n')
        
        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Preference Parameters} \\\\ \n')
        f.write('a0 & ' + '{:.2f}'.format(a0) + ' \\\\ \n')
        f.write('a1 & ' + '{:.2f}'.format(a1) + ' \\\\ \n')
        f.write('a2 & ' + '{:.2f}'.format(a2) + ' \\\\ \n')
        f.write('a3 & ' + '{:.2f}'.format(a3) + ' \\\\ \n')
        f.write('k & ' + '{:.2f}'.format(k) + ' \\\\ \n')
        f.write('gamma & ' + '{:.2f}'.format(gamma) + ' \\\\ \n')
        f.write('lam & ' + '{:.2f}'.format(lam) + ' \\\\ \n')
        f.write('kappa & ' + '{:.2f}'.format(kappa) + ' \\\\ \n')
        f.write('beta\\_yearly & ' + '{:.2f}'.format(beta_yearly) + ' \\\\ \n')
        f.write('delta\\_yearly & ' + '{:.2f}'.format(delta_yearly) + ' \\\\ \n')

        f.write('\\bottomrule \n \n')
        f.write('\\end{tabular} \n')
        f.write('\\end{table} \n')
    
def ttw_table(logdir,logfile,steps,D_nonempAgg,D_nonempAgg_mfx,D_nonempucAgg,D_nonempucAgg_mfx,transObs_E_UIAgg,transObs_E_UIAgg_mfx,emp1924_D_nonemp,emp1924_trans_E_to_UI):
    
    D_nonempAgg_m = D_nonempAgg*12/steps
    D_nonempAgg_mfx_m = D_nonempAgg_mfx*12/steps

    D_nonempucAgg_m = D_nonempucAgg*12/steps
    D_nonempucAgg_mfx_m = D_nonempucAgg_mfx*12/steps
    
    # Marginal effects (normalized to a one-month increase): 
    dDdP = (D_nonempAgg_mfx_m - D_nonempAgg_m)*(steps/12)
    dgdP = (transObs_E_UIAgg_mfx - transObs_E_UIAgg)*(steps/12)

    # Intensive - extensive margin: 
    int_marg = transObs_E_UIAgg_mfx @ dDdP
    ext_marg = D_nonempAgg_m @ dgdP

    # Intensive - extensive margin v2: 
    int_marg2 = transObs_E_UIAgg @ dDdP
    ext_marg2 = D_nonempAgg_mfx_m @ dgdP

    # Full effect:
    dTudP = int_marg + ext_marg

    # Full effect v2:
    dTudP_2 = int_marg2 + ext_marg2

    # Check of full effect by calculating total time out of work separately
    T_U = transObs_E_UIAgg @ D_nonempAgg_m
    T_U_mfx = transObs_E_UIAgg_mfx @ D_nonempAgg_mfx_m
    dTudP_3 = (T_U_mfx - T_U)*(steps/12)

    # 2020_06_01 update: Johannes' midpoint decomposition
    int_marg_mp = ((transObs_E_UIAgg + transObs_E_UIAgg_mfx)/2) @ dDdP
    ext_marg_mp = (D_nonempAgg_m + D_nonempAgg_mfx_m)/2 @ dgdP

    # Full effect, midpoint version:
    dTudP_mp = int_marg_mp + ext_marg_mp

    # T^U empirical
    emp1924_T_U = emp1924_trans_E_to_UI @ emp1924_D_nonemp

    # 2020_06_05 update: calculate T^U and decomposition using uncapped nonemployment durations
    dDucdP = (D_nonempucAgg_mfx_m - D_nonempucAgg_m)*(steps/12)
    int_marg_uc = ((transObs_E_UIAgg + transObs_E_UIAgg_mfx)/2) @ dDucdP
    ext_marg_uc = (D_nonempucAgg_m + D_nonempucAgg_mfx_m)/2 @ dgdP
    dTudP_uc = int_marg_uc + ext_marg_uc
    T_U_uc = transObs_E_UIAgg @ D_nonempucAgg_m
    T_U_uc_mfx = transObs_E_UIAgg_mfx @ D_nonempucAgg_mfx_m
    dTudP_uc_2 = (T_U_uc_mfx - T_U_uc)*(steps/12)
       
    with open(logdir+logfile,'a') as f:
        f.write('\\begin{table}[H]  \centering')
        f.write('\\caption{Estimates for Total Time out of Work } \n')
        f.write('\\begin{tabular}{lc} \\\\ \n')
        f.write('\\toprule \n \n')
        f.write('Estimate & Value (months) \\\\')
        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Empirical Total Time out of Work} \\\\ \n')
        f.write('$T^U$ (empirical) & ' + '{:.6f}'.format(emp1924_T_U) + ' \\\\ \n')
        
        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Simulated Total Time out of Work} \\\\ \n')
        
        f.write('$T^U$ (P=12) & ' + '{:.6f}'.format(T_U) + ' \\\\ \n')
        f.write('$T^U$ (P=15) & ' + '{:.6f}'.format(T_U_mfx) + ' \\\\ \n')
        f.write('d$T^U$/dP & ' + '{:.6f}'.format(dTudP_3) + ' \\\\ \n')
                        
        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Calculate using Decomposition (version 1)} \\\\ \n')
        f.write('Intensive Margin & ' + '{:.6f}'.format(int_marg) + ' \\\\ \n')
        f.write('Extensive Margin & ' + '{:.6f}'.format(ext_marg) + ' \\\\ \n')
        f.write('d$T^U$/dP & ' + '{:.6f}'.format(dTudP) + ' \\\\ \n')

        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Calculate using Decomposition (version 2)} \\\\ \n')
        f.write('Intensive Margin & ' + '{:.6f}'.format(int_marg2) + ' \\\\ \n')
        f.write('Extensive Margin & ' + '{:.6f}'.format(ext_marg2) + ' \\\\ \n')
        f.write('d$T^U$/dP & ' + '{:.6f}'.format(dTudP_2) + ' \\\\ \n')

        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Calculate using Mid-Point Decomposition} \\\\ \n')
        f.write('Intensive Margin & ' + '{:.6f}'.format(int_marg_mp) + ' \\\\ \n')
        f.write('Extensive Margin & ' + '{:.6f}'.format(ext_marg_mp) + ' \\\\ \n')
        f.write('d$T^U$/dP & ' + '{:.6f}'.format(dTudP_mp) + ' \\\\ \n')

        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Simulated Total Time out of Work (uncapped nonemp. durations)} \\\\ \n')
        
        f.write('$T^U$ (P=12) & ' + '{:.6f}'.format(T_U_uc) + ' \\\\ \n')
        f.write('$T^U$ (P=15) & ' + '{:.6f}'.format(T_U_uc_mfx) + ' \\\\ \n')
        f.write('d$T^U$/dP (uncapped) & ' + '{:.6f}'.format(dTudP_uc_2) + ' \\\\ \n')

        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Calculate using Mid-Point Decomposition (uncapped nonemp. durations)} \\\\ \n')
        f.write('Intensive Margin & ' + '{:.6f}'.format(int_marg_uc) + ' \\\\ \n')
        f.write('Extensive Margin & ' + '{:.6f}'.format(ext_marg_uc) + ' \\\\ \n')
        f.write('d$T^U$/dP & ' + '{:.6f}'.format(dTudP_uc) + ' \\\\ \n')

        f.write('\\midrule \n \n')
        f.write('\\multicolumn{2}{l}{Model Fitting} \\\\ \n')
        f.write('dNonemp/dP at age 55 & ' + '{:.6f}'.format(dDdP[20]) + ' \\\\ \n')
        
        f.write('\\bottomrule \n \n')
        
        f.write('\\multicolumn{2}{p{0.7\linewidth}}{\small{Decomposition notes: Version 1 corresponds \
            to: $$\Delta T^u = [g_t(P+1) - g_t(P)]D_t(P) + [D_t(P+1) - D_t(P)]g_t(P+1)$$ \
                Version 2 corresponds \
            to: $$\Delta T^u = [g_t(P+1) - g_t(P)]D_t(P+1) + [D_t(P+1) - D_t(P)]g_t(P)$$ \
            Mid-point decomposition corresponds \
            to: $$\Delta T^u = [g_t(P+1) - g_t(P)]\\frac{D_t(P+1)+D_t(P)}{2} + [D_t(P+1) - D_t(P)]\\frac{g_t(P+1)+g_t(P)}{2}$$ }} \n \n')

        f.write('\\end{tabular} \n')
        f.write('\\end{table} \n')


def writeln(logdir,logfile,line):
    with open(logdir+logfile,'a') as f:
        f.write(line+'\n')
        
        

if __name__ == "__main__":

    d = './'
    f = 'testfile.tex'
#    os.remove(f)
    writeln(d,f,'hello world')
    writeln(d,f,'it is me again')
    
    f = './testfile1.tex'
    latex_header(d,f,'Output File')
    latex_close(d,f)
    
    


