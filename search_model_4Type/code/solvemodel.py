import sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import estimagic as em
from numba import njit


momentsfile = "../data/base_moments_Germany_wages.xlsx"

def mu(xi, t):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi
    return mu_S + pi * np.minimum(t - kappa, np.zeros(len(t)))


def predictedMoments(xi, b, s, logphi):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi
    lastperiod = len(b)
    muv = mu(xi, np.arange(1, lastperiod + 1))
    haz = np.zeros(len(b))
    logw_reemp = np.zeros(len(b))

    for t in range(lastperiod - 2, -1, -1):
        omega = (logphi[t + 1] - muv[t]) / sigma
        if omega >= 7:
            omega = 7
        haz[t] = s[t] * (1 - norm.cdf(omega))
        logw_reemp[t] = muv[t] + sigma * norm.pdf(omega) / (
            1 - norm.cdf(omega)
        )

    haz[-1] = haz[-2]
    logw_reemp[-1] = logw_reemp[-2]

    return haz, logw_reemp


def optimalPath(xi, b):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi
    lastperiod = len(b)
    muv = mu(xi, np.arange(1, lastperiod + 1))
    s = np.zeros(lastperiod)
    logphi = np.zeros(lastperiod)

    # Assuming steadyState is another function to compute steady state
    s[-1], logphi[-1] = steadyState(xi, b[-1])

    for t in range(lastperiod - 2, -1, -1):
        omega = (logphi[t + 1] - muv[t + 1]) / sigma
        if omega > 7:
            omega = 7
        integral = (1 - norm.cdf(omega)) * (
            muv[t + 1]
            - logphi[t + 1]
            + sigma
            * norm.pdf(omega)
            / (1 - norm.cdf(omega))
        )
        if np.isnan(integral) or integral < 0:
            integral = 0
        s[t] = min((1 / k * delta / (1 - delta) * integral) ** (1 / gamma), 1)
        logphi[t] = (
            (1 - delta) * (np.log(b[t]) - k * (s[t] ** (1 + gamma)) / (1 + gamma))
            + delta * logphi[t + 1]
            + delta * s[t] * integral
        )

    return s, logphi

# @njit(cache=True)
def clip_gradient(grad, max_grad=1):
    norm = np.linalg.norm(grad)
    if norm > max_grad:
        grad = grad / norm * max_grad
    return grad

# @njit(cache=True)
def check_and_update_bound(func, x, bounds):
    for j, (lower, upper) in enumerate(bounds):
        if lower is not None:
            x_at_lower = np.array(x)
            x_at_lower[j] = lower
            if func(x_at_lower) < func(x):
                x[j] = lower
        if upper is not None:
            x_at_upper = np.array(x)
            x_at_upper[j] = upper
            if func(x_at_upper) < func(x):
                x[j] = upper
    return x


def steadyState(xi, b_S):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi

    # @njit(cache=True)
    def steadyStateSystem(x):
        s, q = x
        omega = (q - mu_S) / sigma
        if omega > 7:
            omega = 7
        integral = (1 - norm.cdf(omega)) * (
            mu_S
            - q
            + sigma * norm.pdf(omega) / (1 - norm.cdf(omega))
        )
        if np.isnan(integral) or integral < 0:
            integral = 0

        # FOC for search effort
        # f1 = min(s, 1) - (1 / k * delta / (1 - delta) * integral) ** (1 / gamma)
        f1 = s - (1 / k * delta / (1 - delta) * integral) ** (1 / gamma)
        # FOC for reservation wage
        f2 = (
            -q
            + np.log(b_S)
            - k * (min(s, 1) ** (1 + gamma)) / (1 + gamma)
            + delta / (1 - delta) * min(s, 1) * integral
        )

        # Return sum of squares of the deviations
        return f1**2 + f2**2

    # Bounds for s and q
    # bounds = [(0, 1), (0, None)]  # s between 0 and 1, q greater than 0
    bounds = [(0, None), (0, None)]  # s between 0 and 1, q greater than 0

    # Initial guess
    x0 = [0.05, mu_S]

    # Solve the system of equations
    res = minimize(steadyStateSystem, x0, bounds=bounds, method="L-BFGS-B")
    s_S = res.x[0]
    logphi_S = res.x[1]


    s_S_cap = min(s_S,1)

    return s_S_cap, logphi_S


def solveModel(xi, institutions):
    T, b = institutions

    s, logphi = optimalPath(xi, b)
    haz, logw_reemp = predictedMoments(xi, b, s, logphi)
    surv = np.ones(len(b))
    for t in range(1, len(b)):
        surv[t] = surv[t - 1] * (1 - haz[t - 1])
    # T=96
    
    Tminb = T - len(b)
    haz_long = np.concatenate((haz, np.ones(Tminb) * haz[-1]))
    logw_reemp_long = np.concatenate((logw_reemp, np.ones(Tminb) * logw_reemp[-1]))

    surv_long = np.ones(T)
    for t in range(1, T):
        surv_long[t] = surv_long[t - 1] * (1 - haz_long[t - 1])

    D = np.sum(surv_long)
    dens_long = haz_long * surv_long
    if np.sum(dens_long)!=0:
        E_logw_reemp = np.sum(dens_long * logw_reemp_long) / np.sum(dens_long)
    else: 
        E_logw_reemp = 0

    # E_logw_reemp = 0
    return s, logphi, haz, logw_reemp, surv, D, E_logw_reemp


# UI benefit level b as function of time
def benefit_path(b1, b2, b3, welfare, T1, T2, T3, T):
    """
    Returns the benefit path given parameter values.
        Arguments:
            b1,b2,b3,welfare (all floats): values at distinct time windows during unemployment
            T1,T2,T3 (int): points in Unemp. spell when benefit levels change.
            T (int): Total number of periods
        Returns:
            benefits (array): Benefit path in unemployment
    """
    benefits = np.zeros(T)
    benefits[0:T1] = b1
    benefits[T1:T2] = b2
    benefits[T2:T3] = b3
    benefits[T3:T] = welfare

    return benefits


# function to solve individual labor supply model
def solveSingleTypeModel(xi, institutions):
    """
    Solves the model for a single-type individual.
        Arguments:
            params (array): Array of structural parameters.
            institutions (array): Array of institional parameters
        Returns:
            valOLF (array): Value function of OLF

    """

    s, logphi, haz, logw_reemp, surv, D, E_logw_reemp = solveModel(xi, institutions)

    return s, logphi, haz, logw_reemp, surv, D, E_logw_reemp


def solveMultiTypeModel(params, institutions):
    """
    Solves the  model in a multi-type setup.
        Arguments:
            params (array): Array of parameter values for all types.
        Returns:
    """

    delta, k1, gamma, mu1, sigma, kappa, pi, k2, k3, k4, mu2, mu3, mu4, q2, q3, q4 = params

    # Parameters for single type model
    xi = np.copy(params[0:7])

    # Variables for 2-type estimation
    q1 = 1 - q2 - q3 - q4

    # Type 1
    s1, logphi1, haz1, w_reemp1, surv1, D1, E_w_reemp1 = solveModel(xi, institutions)

    # Type 2
    xi[1] = k2  # Adjust the k parameter for Type 2
    xi[3] = mu2
    s2, logphi2, haz2, w_reemp2, surv2, D2, E_w_reemp2 = solveModel(
        xi, institutions
    )

    # Type 3
    xi[1] = k3  # Adjust the k parameter for Type 2
    xi[3] = mu3
    s3, logphi3, haz3, w_reemp3, surv3, D3, E_w_reemp3 = solveModel(
        xi, institutions
    )

    # Type 4
    xi[1] = k4  # Adjust the k parameter for Type 2
    xi[3] = mu4
    s4, logphi4, haz4, w_reemp4, surv4, D4, E_w_reemp4 = solveModel(
        xi, institutions
    )

    # Aggregate Survival
    survival = q1 * surv1 + q2 * surv2 + q3 * surv3 + q4 * surv4

    # Calculate share of each type left at beginning in unemployment
    # weight1 = q1 * surv1 / survival
    # weight2 = q2 * surv2 / survival
    # weight3 = q3 * surv2 / survival
    # Create a mask for elements in 'survival' that are not equal to 0
    mask = survival != 0

    # Initialize 'weight1' with zeros
    weight1 = np.zeros_like(survival, dtype=float)
    weight2 = np.zeros_like(survival, dtype=float)
    weight3 = np.zeros_like(survival, dtype=float)
    weight4 = np.zeros_like(survival, dtype=float)

    # Perform the division only where 'mask' is True
    weight1[mask] = q1 * surv1[mask] / survival[mask]
    weight2[mask] = q2 * surv2[mask] / survival[mask]
    weight3[mask] = q3 * surv3[mask] / survival[mask]
    weight4[mask] = q4 * surv4[mask] / survival[mask]
    

    # Calculate aggregate hazard and average reemployment wage of leavers
    haz_agg = weight1 * haz1 + weight2 * haz2 + weight3 * haz3 + weight4 * haz4
    if min(haz_agg)==0:
        w_reemp_agg = (weight1 * w_reemp1 + weight2  * w_reemp2 + weight3  * w_reemp3 + weight4  * w_reemp4) 
    else:
        w_reemp_agg = (weight1 * haz1 * w_reemp1 + weight2 * haz2 * w_reemp2 + weight3 * haz3 * w_reemp3 + weight4 * haz4 * w_reemp4) / haz_agg

    # Aggregate expected duration
    D = q1 * D1 + q2 * D2 + q3 * D3 + q4 * D4

    # Aggregate expected wage
    W = q1 * E_w_reemp1 + q2 * E_w_reemp2 + q3 * E_w_reemp3 + q4 * E_w_reemp4

    return haz_agg, w_reemp_agg, survival, D, W

def simulate_moments(params, institutions_pre, institutions_post):

    # Simulate Model
    haz_agg1, w_reemp_agg1, survival1, D1, W1 = solveMultiTypeModel(
        params, institutions_pre
    )

    haz_agg2, w_reemp_agg2, survival2, D2, W2 = solveMultiTypeModel(
        params, institutions_post
    )

    # Return Moments
    dDdP = (D2-D1)/6 
    dWdP = (W2-W1)/6

    moments_model = np.hstack((haz_agg1[:30], 
                                haz_agg2[:30],
                                w_reemp_agg1[:24], 
                                w_reemp_agg2[:24],
                                dDdP,
                                dWdP
                                ))

    return moments_model

def sse(params, target, W, institutions_pre, institutions_post):
    simmoments = simulate_moments(params, institutions_pre, institutions_post)

    # Deviations between target moments and simulated moments:
    err = target - simmoments
    # Calculate SSE
    SSEval = err.T @ W @ err

    return SSEval


def matchingMoments():

    moments_df = pd.read_excel(momentsfile, index_col=0)
    moments_hazard_12 = moments_df["haz12"].to_numpy()[1:]
    moments_hazard_18 = moments_df["haz18"].to_numpy()[1:]
    moments_wage_12 = moments_df["wage12"].to_numpy()[1:25]
    moments_wage_18 = moments_df["wage18"].to_numpy()[1:25]

    D12_true=14.225
    D18_true=15.175
    B12_true=6.685
    B18_true=8.455
    LogPostWage12_true=4.0139
    LogPostWage18_true=4.0061
    dDdP_true = 0.16
    dWdP_true = -0.0013

    target = np.hstack((moments_hazard_12, 
                        moments_hazard_18,
                        moments_wage_12,
                        moments_wage_18,
                        dDdP_true,
                        dWdP_true))
    

    # Covariance Matrix:
    se_haz12 = moments_df["haz12_se"].to_numpy()[1:]
    se_haz18 = moments_df["haz18_se"].to_numpy()[1:]
    var_haz12 = se_haz12**2
    var_haz18 = se_haz18**2
    se_wage12 = moments_df["wage12_se"].to_numpy()[1:25] * .1
    se_wage18 = moments_df["wage18_se"].to_numpy()[1:25] * .1
    var_wage12 = se_wage12**2
    var_wage18 = se_wage18**2
    se_dDdP = 0.1 * .01
    se_dWdP = 0.3 * .01
    var = np.hstack((var_haz12, 
                     var_haz18,
                     var_wage12,
                     var_wage18,
                     se_dDdP**2,
                     se_dWdP**2))
    cov = np.eye(len(var)) * var

    return target, cov, moments_hazard_12, moments_hazard_18, moments_wage_12, moments_wage_18

class gmm:
    def __init__(
        self, params_full, target, W, institutions_pre, institutions_post, disp=False
    ):
        self.iter = 0
        self.params_full = params_full
        self.target = target
        self.W = W
        self.institutions_pre = institutions_pre
        self.institutions_post = institutions_post
        self.disp = disp

    def sse(self, params):
        # Deviations between target moments and simulated moments:
        self.params_full.update(params)
        params_full_vec = np.array(self.params_full["value"])
        simmoments = simulate_moments(
            params_full_vec, self.institutions_pre, self.institutions_post
        )

        # Deviations between target moments and simulated moments:
        err = self.target - simmoments
        # Calculate SSE
        SSEval = err.T @ self.W @ err

        self.iter = self.iter + 1
        if self.disp:
            print("Iter: {:.0f}; Current SSE: {:10.3f}".format(self.iter, SSEval))

        return SSEval

    def criterion(self, params):
        # Deviations between target moments and simulated moments:
        self.params_full.update(params)
        params_full_vec = np.array(self.params_full["value"])
        simmoments = simulate_moments(
            params_full_vec, self.institutions_pre, self.institutions_post
        )

        # Deviations between target moments and simulated moments:
        err = self.target - simmoments
        # Calculate SSE
        # SSEval = err.T @ self.W @ err

        L = np.linalg.cholesky(self.W)
        weighted_residuals = err @ L

        weighted_residuals_squared = weighted_residuals**2
        SSEval = weighted_residuals_squared.sum()
        out = {
            # root_contributions are the least squares residuals.
            # if you square and sum them, you get the criterion value
            "root_contributions": weighted_residuals,
            # if you sum up contributions, you get the criterion value
            "contributions": weighted_residuals_squared,
            # this is the standard output
            "value": SSEval,
        }
        self.iter = self.iter + 1
        if self.disp:
            print("Iter: {:.0f}; Current SSE: {:10.3f}".format(self.iter, SSEval))

        return out


if __name__ == "__main__":

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
        # === 4 Estimating All parameters ===

        # --- 4.1 GMM Class ---
        delta = 0.995 
        # k = 20 
        gamma = 0.145 
        mu1 = 3.
        sigma = 0.5
        kappa = 12 
        pi = 0
        k1 = 10
        k2 = 1
        k3 = 0
        q2 = .5
        q3 = 0
        mu2 = 2.5
        mu3 = 2
        params_full = pd.DataFrame(
            data={
                "value": [delta, k1, gamma, mu1, sigma, kappa, pi,k2,k3,mu2,mu3,q2,q3]
            },
            index=["delta", "k1", "gamma", "mu1", "sigma", "kappa", "pi", "k2", "k3", "mu2", "mu3", "q2", "q3"]
    ,
        )
        gmm_object = gmm(params_full.copy(),target, W, inst1, inst2, disp=True)

        params = pd.DataFrame(
            data={
            "value"       : [20], "lower_bound" : [0.01],  "upper_bound" : [2000]
            },
            index=["k1"],
        )
        print(gmm_object.sse(params))

    if 1:
 

        # --- 4.2 Estimate gamma using Estimagic ---
        # algo = "tao_pounders"
        print('\n === Estimation using Estimagic === \n')
        res = em.minimize(
            criterion=gmm_object.criterion,
            # criterion=gmm_object.sse,
            params=params,
            algorithm=algo,
        )

        print('Estimated Parameters:')
        print(res.params)
        print(res.criterion)
        print(res)
        col0 = res.params['value']
        col0.loc['SSE'] = res.criterion
        em.criterion_plot(res,monotone=True)

        print(gmm_object.params_full)
        gmm_object.params_full.update(res.params)
        
        # gmm_object = gmm(params_full.copy(),target, W, inst1, inst2, disp=False)
        
        print(gmm_object.params_full)

        params_result=np.array(gmm_object.params_full['value'])

        h1,w1,S1,D1,W1 = solveMultiTypeModel(params_result,inst1)
        h2,w2,S2,D2,W2 = solveMultiTypeModel(params_result,inst2)


        plt.clf()
        plt.plot(timevec, h1, label="Hazard, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, h2, label="Hazard, P=18", linestyle="dashed", color=red)
        plt.plot(timevec[:30], target_h12, label="Moments Hazard, P=12", linestyle="solid", color=blue)
        plt.plot(timevec[:30], target_h18, label="Moments Hazard, P=18", linestyle="solid", color=red)
        plt.title("Exit Hazard")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_haz.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

        plt.clf()
        plt.plot(timevec[:24], w1[:24], label="Wage, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec[:24], w2[:24], label="Wage, P=18", linestyle="dashed", color=red)
        plt.plot(timevec[:24], target_w12[:24], label="Moments Wage, P=12", linestyle="solid", color=blue)
        plt.plot(timevec[:24], target_w18[:24], label="Moments Wage, P=18", linestyle="solid", color=red)
        plt.title("Reemployment Wage")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

    if 0:
        # algo = "tao_pounders"
        algo = "scipy_lbfgsb"
        gmm_object = gmm(params_full.copy(),target, W, inst1, inst2, disp=False)
        # --- Multistart --- 
        res = em.minimize(
            criterion=gmm_object.criterion,
            params=params,
            algorithm=algo,
            multistart=True,
            multistart_options = {
                "n_cores"  : 4,
                "n_samples": 40,
                "sampling_method" : 'latin_hypercube',
                "convergence_max_discoveries" : 2,
                "share_optimizations" : .1
            }
        )

        print('Estimated Parameters:')
        print(res.params)
        print(res.criterion)
        print(res)
        col0 = res.params['value']
        col0.loc['SSE'] = res.criterion
        em.criterion_plot(res,monotone=True)

        print(gmm_object.params_full)
        gmm_object.params_full.update(res.params)
        
        # gmm_object = gmm(params_full.copy(),target, W, inst1, inst2, disp=False)
        
        print(gmm_object.params_full)

        params_result=np.array(gmm_object.params_full['value'])

        h1,w1,S1,D1 = solveMultiTypeModel(params_result,inst1)
        h2,w2,S2,D2 = solveMultiTypeModel(params_result,inst2)


        plt.clf()
        plt.plot(timevec, h1, label="Hazard, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, h2, label="Hazard, P=18", linestyle="dashed", color=red)
        plt.plot(timevec[:30], target_h12, label="Moments Hazard, P=12", linestyle="solid", color=blue)
        plt.plot(timevec[:30], target_h18, label="Moments Hazard, P=18", linestyle="solid", color=red)
        plt.title("Exit Hazard")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_haz.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

        plt.clf()
        plt.plot(timevec[:25], w1[:25], label="Wage, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec[:25], w2[:25], label="Wage, P=18", linestyle="dashed", color=red)
        plt.plot(timevec[:25], target_w12[:25], label="Moments Wage, P=12", linestyle="solid", color=blue)
        plt.plot(timevec[:25], target_w18[:25], label="Moments Wage, P=18", linestyle="solid", color=red)
        plt.title("Reemployment Wage")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

    if 0:

        # Two UI regimes
        # b1 = np.concatenate((np.ones(12) * 190, np.ones(12) * 90))
        # b2 = np.concatenate((np.ones(18) * 190, np.ones(6) * 90))

        # lastperiod = len(b1)

        # Model Parameters
        # xi = [0.995, 150, 0.145, 5.995, 0.5, 12, 0]
        # T = 24
        # P1 = 12
        # P2 = 18
        # ben1 = np.ones(T) * 90
        # ben2 = np.ones(T) * 90
        # ben1[:P1] = 190
        # ben2[:P2] = 190
        # inst1 = T, b1
        # inst2 = T, b2

        # # Solve the model for both benefit regimes
        # s1, logphi1, haz1, logw1, surv1, D1, Ew1 = solveModel(xi, inst1)
        # s2, logphi2, haz2, logw2, surv2, D2, Ew2 = solveModel(xi, inst2)

        # # s1, logphi1, haz1, logw1, surv1, D1, Ew1 = solveModel1(xi, b1)
        # # s2, logphi2, haz2, logw2, surv2, D2, Ew2 = solveModel1(xi, b2)

        # # Plotting
        # fontsize = 16
        # time = np.arange(1, lastperiod + 1)

        # # UI Benefit Paths
        # plt.figure(figsize=(8, 6))
        # plt.plot(time, b1, "r", label="P=12")
        # plt.plot(time, b2, "b", label="P=18")
        # plt.legend(fontsize=fontsize)
        # plt.xlabel("Time", fontsize=fontsize)
        # plt.title("UI Benefit Paths", fontsize=fontsize + 2)
        # plt.ylim(0, 250)  # Adjusted y-axis limit
        # plt.savefig("./log/fig21_bpath.pdf")
        # plt.show()

        # # Search Effort
        # plt.figure(figsize=(8, 6))
        # plt.plot(time, s1, "r", label="P=12")
        # plt.plot(time, s2, "b", label="P=18")
        # plt.legend(fontsize=fontsize)
        # plt.xlabel("Time", fontsize=fontsize)
        # plt.title("Search Effort", fontsize=fontsize + 2)
        # plt.ylim(0, 0.09)  # Adjusted y-axis limit
        # plt.savefig("./log/fig21_s.pdf")
        # plt.show()

        # 3.1 Standard Model 1 type

        delta = 0.995 
        k = 20 
        gamma = 0.145 
        mu1 = 5.995
        sigma = 0.5
        kappa = 12 
        pi = 0
        xi = [delta, k, gamma, mu1, sigma, kappa, pi ]
        # xi = [0.995, 150, 0.145, 5.995, 0.5, 12, 0]

        params = np.array([delta, k, gamma, mu1, sigma, kappa, pi])

        # s1, logphi1, haz1, logw1, surv1, D1, Ew1 = solveSingleTypeModel(params, inst1)
        # s2, logphi2, haz2, logw2, surv2, D2, Ew2 = solveSingleTypeModel(params, inst2)


        # plt.clf()
        # plt.plot(timevec, s1, label="Hazard, P=12", linestyle="dashed", color=blue)
        # plt.plot(timevec, s2, label="Hazard, P=18", linestyle="dashed", color=red)
        # plt.title("Exit Hazard")
        # plt.xlabel("Months")
        # plt.legend(loc="lower left")
        # plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.10)
        # plt.savefig("./log/fig_31.pdf", bbox_inches="tight")
        # plt.show()
        # plt.clf()

        # plt.clf()
        # plt.plot(timevec[:25], logw1[:25], label="Wage, P=12", linestyle="dashed", color=blue)
        # plt.plot(timevec[:25], logw2[:25], label="Wage, P=18", linestyle="dashed", color=red)
        # plt.title("Reemployment Wage")
        # plt.xlabel("Months")
        # plt.legend(loc="lower left")
        # plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # # plt.ylim(bottom=0, top=0.15)
        # plt.savefig("./log/fig_35c.pdf", bbox_inches="tight")
        # plt.show()
        # plt.clf()

        #  Multi Type Aggregate

        k1 = 20
        k2 = 10
        k3 = 0
        mu2 = 2.5
        mu3 = 2
        q1 = 1
        q2 = 0
        paramsMulti = np.array([delta, k1, gamma, mu1, sigma, kappa, pi,k2,k3,mu2,mu3,q1,q2])

        # P=12 group
        haz_agg1, w_reemp_agg1, survival1, D1  = solveMultiTypeModel(paramsMulti, inst1)

        # P=18 group
        haz_agg2, w_reemp_agg2, survival2, D2 = solveMultiTypeModel(
            paramsMulti, inst2
        )

        plt.clf()
        plt.plot(timevec, haz_agg1, label="Hazard, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, haz_agg2, label="Hazard, P=18", linestyle="dashed", color=red)
        plt.title("Exit Hazard")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_35c.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

        plt.clf()
        plt.plot(timevec, survival1, label="Survival, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, survival2, label="Survival, P=18", linestyle="dashed", color=red)
        plt.title("Survival in Unemployment")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_35c.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

        plt.clf()
        plt.plot(timevec[:25], w_reemp_agg1[:25], label="Survival, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec[:25], w_reemp_agg2[:25], label="Survival, P=18", linestyle="dashed", color=red)
        plt.title("Reemployment Wage")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_35c.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

    