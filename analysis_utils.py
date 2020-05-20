import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.manifold import TSNE
import scipy.optimize as opt
import piscola as pisco

import lmfit
import emcee
import corner
from IPython.display import display, Math
from itertools import product

from astropy.cosmology import FlatLambdaCDM
Omega_m = 0.3
H0 = 70.0
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

def decompose(algorithm, features, phases, weights=None, n_components=3, plot=True, save_plot=False, name_plot='decomposition.png'):
    
    kwargs = {}
    if (weights is not None) and ('sklearn' not in str(algorithm)) and ('umap' not in str(algorithm)):
        kwargs = {'weights':weights}
        
    if ('sklearn' in str(algorithm)) or ('wpca' in str(algorithm)) or ('umap' in str(algorithm)):
        reducer = algorithm(n_components=n_components)
        transformed_features = reducer.fit_transform(features, **kwargs) # fits AND transforms at the same time
        try:
            components_ = reducer.components_
        except:
            pass
    else:
        reducer = algorithm(feature_values, n_components=n_components, V=weights)
        reducer.SolveNMF()
        transformed_features = reducer.W
        components_ = reducer.H

    color_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]
    color_palette = color_palette[::3]

    if ('umap' not in  str(algorithm)) and ('TSNE' not in  str(algorithm)) and plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in np.arange(n_components):
            variance = transformed_features[:,i].var()/np.sum(transformed_features.var(axis=0))
            component_array = components_[i,:]
            ax.plot(phases, component_array, 'o-', 
                    label = f'NMF comp. {i+1} ({int(np.round(100*variance, 0))}%)', color=color_palette[i])

        ax.set_ylabel('Component value', fontsize=20)
        ax.set_xlabel('Days with respect to B-band peak', fontsize=20)
        ax.tick_params('both', labelsize=20)
        ax.legend(fontsize=14)
        if save_plot:
            plt.savefig(f'{name_plot}')
        plt.show()
    
    return reducer, transformed_features

def monte_carlo_decomposition(feature_values, dfeature_values, phases, n_components=3, N=1000,
                              plot=True, save_plot=False, name_plot='decomposition.png'):
    
    #iterate in SN
    sample_flux = []
    for lc, lcerr in zip(feature_values, dfeature_values):

        # iterate in epoch
        sn_flux = np.asarray([np.random.normal(flux, err, N) for flux, err in zip(lc, lcerr)])
        sample_flux.append(sn_flux)  # append SN light curve
        
    sample_flux = np.asarray(sample_flux)
    sample_flux[sample_flux < 0] = 0  # remove negative values
    
    # decompose
    transformed_features_list = []
    for i in range(N):
        features = sample_flux[:,:,i]
        reducer, transformed_features = decompose(NMF, features, phases, weights=1/dfeature_values**2, n_components=n_components, plot=False)
        transformed_features_list.append(transformed_features)
    
    transformed_features_array = np.asarray(transformed_features_list)
    transformed_features_array = np.swapaxes(transformed_features_array,0,1)
    transformed_features = np.asarray([np.mean(array, axis=0) for array in transformed_features_array])  # mean values
    dtransformed_features = np.asarray([np.std(array, axis=0) for array in transformed_features_array])  # uncertainties/errors
    
    # covariance
    ctransformed_features = []
    for array, sn_flux in zip(transformed_features_array, sample_flux):
        covariances = []
        for i, j in product(range(n_components), range(n_components)):
            if j <= i:
                continue
            if i==0:
                # the first parameters is going to be mb (flux at phase 0.0) instead of p1
                array[:, i] = sn_flux[np.argmin(np.abs(phases-0.0))]
            cov_matrix = np.cov(array[:, i], array[:, j])
            covariances.append(cov_matrix[0][1])  # choose one of the non-diagonals

        ctransformed_features.append(covariances)
    ctransformed_features = np.asarray(ctransformed_features)
    
    # for plotting purposes
    reducer, transformed_features, decompose(NMF, feature_values, phases, weights=1/dfeature_values**2, 
                                              n_components=n_components, plot=plot, save_plot=save_plot,
                                              name_plot=name_plot)
    
    return reducer, (transformed_features, dtransformed_features, ctransformed_features)

    
def salt2_residual(params, mb, x1, color, dmb, dx1, dcolor, z, dz=0.0, 
                   log_mass=0.0, dlog_mass=0.0, gamma=0.0,
                   sig_int=0.0, sig_lensing=0.0, sig_hostcorr=0.0, 
                   cov_m_s=0.0, cov_m_c=0.0, cov_s_c=0.0):
    
        try:
            alpha = params['alpha'].value
        except:
            alpha = 0.0

        try:
            M = params['M'].value
        except:
            M = 0.0

        try:
            beta = params['beta'].value
        except:
            beta = 0.0

        try:
            gamma = np.asarray([params['gamma'].value if mass >= 10 
                                else 0.0 for mass in log_mass])
        except:
            pass

        try:
            Omega_m = params['Omega_m'].value
            H0 = 70
            mu_cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m).distmod(z).value
        except:
            mu_cosmo = cosmo.distmod(z).value

        mu_SN = mb - (M + gamma) + alpha*x1 - beta*color

        D = (dmb**2 + (alpha**2)*(dx1**2) + (beta**2)*(dcolor**2)
                + (5*dz/(z*np.log(10)))**2
                + sig_lensing**2 + sig_hostcorr**2 + sig_int**2
                + 2*alpha*cov_m_s + 2*(-beta)*cov_m_c + 2*alpha*(-beta)*cov_s_c
               )

        Nparams = len(params)
        dof = len(mu_SN) - Nparams
        chi2 = np.sum((mu_SN - mu_cosmo)**2/D)

        return chi2

def salt2_mcmc(p0, args, labels=None, nwalkers=32, nsteps=1000, nburn=500, verbose=True, plot=False):

    def log_prior(theta):

        if len(theta) == 2:
            M, alpha = theta
            if -20.0 < M < -18 and 0.0 < alpha < 2.0:
                return 0.0
        
        elif len(theta) == 3:
            M, alpha, beta = theta
            if -20.0 < M < -18 and 0.0 < alpha < 2.0 and 0.0 < beta < 8.0:
                return 0.0
            
        elif len(theta) == 4:
            M, alpha, beta, gamma = theta
            if -20.0 < M < -18 and 0.0 < alpha < 2.0 and 0.0 < beta < 8.0 and -0.2 < gamma < 0.2:
                return 0.0
            
        elif len(theta) == 5:
            M, alpha, beta, gamma, Omega_m = theta
            if -20.0 < M < -18 and 0.0 < alpha < 2.0 and 0.0 < beta < 8.0 and -0.2 < gamma < 0.2 and 0.0 < Omega_m < 1.0:
                return 0.0
            
        return -np.inf

    def log_likelihood(theta, mb, x1, color, dmb, dx1, dcolor, z, dz=0.0, 
                       log_mass=0.0, dlog_mass=0.0, gamma=0.0,
                       sig_int=0.0, sig_lensing=0.0, sig_hostcorr=0.0, 
                       cov_m_s=0.0, cov_m_c=0.0, cov_s_c=0.0):

        if len(theta) == 2:
            M, alpha = theta
            beta = 0.0
            mu_cosmo = cosmo.distmod(z).value
        elif len(theta) == 3:
            M, alpha, beta = theta
            mu_cosmo = cosmo.distmod(z).value
        elif len(theta) == 4:
            M, alpha, beta, gamma = theta
            gamma = np.asarray([gamma if mass >= 10 else 0.0 for mass in log_mass])
            mu_cosmo = cosmo.distmod(z).value
        elif len(theta) == 5:
            M, alpha, beta, gamma, Omega_m = theta
            gamma = np.asarray([gamma if mass >= 10 else 0.0 for mass in log_mass])
            H0 = 70
            mu_cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m).distmod(z).value
            
        mu_SN = mb - (M + gamma) + alpha*x1 - beta*color

        D = (dmb**2 + (alpha**2)*(dx1**2) + (beta**2)*(dcolor**2)
                    + (5*dz/(z*np.log(10)))**2
                    + sig_lensing**2 + sig_hostcorr**2 + sig_int**2
                    + 2*alpha*cov_m_s + 2*(-beta)*cov_m_c + 2*alpha*(-beta)*cov_s_c
                   )
        
        return -0.5*np.sum((mu_SN - mu_cosmo)**2/D + np.log(D))

    def log_probability(theta, mb, x1, color, dmb, dx1, dcolor, z, dz=0.0, 
                       log_mass=0.0, dlog_mass=0.0, gamma=0.0,
                       sig_int=0.0, sig_lensing=0.0, sig_hostcorr=0.0, 
                       cov_m_s=0.0, cov_m_c=0.0, cov_s_c=0.0):

        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, mb, x1, color, dmb, dx1, dcolor, z, dz, 
                                   log_mass, dlog_mass, gamma, sig_int, sig_lensing, sig_hostcorr, 
                                   cov_m_s, cov_m_c, cov_s_c)

    ndim = len(p0)
    pos = p0 + 1e-4*np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=args)
    sampler.run_mcmc(pos, nsteps);
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    
    if plot==True:
        if labels is None:
            labels = [f'param{i}' for i in ndim]
        fig = corner.corner(samples, labels=labels)
        plt.show()
        
    results = list()
    errors = list()
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        results.append(mcmc[1])
        q = np.diff(mcmc)
        errors.append(q)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i].replace('$', ''))
        if verbose==True:
            display(Math(txt))
        
    return results, errors

##########################################################################################

def pisco_residual(params, mb, p2, p3, color, dmb, dp2, dp3, dcolor, z, dz=0.0, 
                   log_mass=0.0, dlog_mass=0.0, gamma=0.0,
                   sig_int=0.0, sig_lensing=0.0, sig_hostcorr=0.0, 
                   cov_m_p2=0.0, cov_m_p3=0.0, cov_m_c=0.0,
                   cov_p2_p3=0.0, cov_p2_c=0.0, cov_p3_c=0.0):
    
    try:
        eta = params['eta'].value
    except:
        eta = 0.0
        
    try:
        M = params['M'].value
    except:
        M = 0.0
    
    try:
        theta = params['theta'].value
    except:
        theta = 0.0
    
    try:
        beta = params['beta'].value
    except:
        beta = 0.0
        
    try:
        gamma = np.asarray([params['gamma'].value if mass >= 10 
                            else 0.0 for mass in log_mass])
    except:
        pass

    try:
        Omega_m = params['Omega_m'].value
        H0 = 70
        mu_cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m).distmod(z).value
    except:
        mu_cosmo = cosmo.distmod(z).value
        
    mu_SN = mb - (M + gamma) + eta*p2 + theta*p3 - beta*color
    
    D = (dmb**2 + (eta**2)*(dp2**2) + (theta**2)*(dp3**2) + (beta**2)*(dcolor**2)
            + (5*dz/(z*np.log(10)))**2
            + sig_lensing**2 + sig_hostcorr**2 + sig_int**2
            + 2*eta*cov_m_p2 + 2*theta*cov_m_p3 + 2*(-beta)*cov_m_c 
            + 2*eta*theta*cov_p2_p3 + 2*eta*(-beta)*cov_p2_c
            + 2*theta*(-beta)*cov_p3_c
           )
    
    Nparams = len(params)
    dof = len(mu_SN) - Nparams
    chi2 = np.sum((mu_SN - mu_cosmo)**2/D)
    
    return chi2

def pisco_mcmc(p0, args, labels=None, nwalkers=32, nsteps=1000, nburn=500, verbose=True, plot=False):

    def log_prior(params):
        
        try:
            i = labels.index('M')
            M = params[i]
            constrain_M = -20.0 < M < -15
        except:
            constrain_M = True
        try:
            i = labels.index(r'$\eta$')
            eta = params[i]
            constrain_eta = -10.0 < eta < 10.0
        except:
            constrain_eta = True
        try:
            i = labels.index(r'$\theta$')
            theta = params[i]
            constrain_theta = 0.0 < theta < 40.0
        except:
            constrain_theta = True
        try:
            i = labels.index(r'$\beta$')
            beta = params[i]
            constrain_beta = 0.0 < beta < 8.0
        except:
            constrain_beta = True
        try:
            i = labels.index(r'$\gamma$')
            gamma = params[i]
            constrain_gamma = -0.2 < gamma < 0.2
        except:
            constrain_gamma = True
        try:
            i = labels.index(r'$\Omega_m$')
            Omega_m = params[i]
            constrain_Omega_m = 0.0 < Omega_m < 1.0
        except:
            constrain_Omega_m = True
        
        if constrain_M and constrain_eta and constrain_theta and constrain_beta and constrain_gamma and constrain_Omega_m:
                return 0.0
            
        return -np.inf

    def log_likelihood(params, mb, p2, p3, color, dmb, dp2, dp3, dcolor, z, dz=0.0, 
                       log_mass=0.0, dlog_mass=0.0, gamma=0.0,
                       sig_int=0.0, sig_lensing=0.0, sig_hostcorr=0.0, 
                       cov_m_p2=0.0, cov_m_p3=0.0, cov_m_c=0.0,
                       cov_p2_p3=0.0, cov_p2_c=0.0, cov_p3_c=0.0):

        
        try:
            i = labels.index('M')
            M = params[i]
        except:
            M = 0.0
        try:
            i = labels.index(r'$\eta$')
            eta = params[i]
        except:
            eta = 0.0
        try:
            i = labels.index(r'$\theta$')
            theta = params[i]
        except:
            theta = 0.0
        try:
            i = labels.index(r'$\beta$')
            beta = params[i]
        except:
            beta = 0.0
        try:
            i = labels.index(r'$\gamma$')
            gamma = np.asarray([params[i] if mass >= 10 else 0.0 for mass in log_mass])
        except:
            pass        
        try:
            i = labels.index(r'$\Omega_m$')
            Omega_m = params[i]
            H0 = 70
            mu_cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m).distmod(z).value
        except:
            mu_cosmo = cosmo.distmod(z).value
            
        mu_SN = mb - (M + gamma) + eta*p2 + theta*p3 - beta*color

        D = (dmb**2 + (eta**2)*(dp2**2) + (theta**2)*(dp3**2) + (beta**2)*(dcolor**2)
            + (5*dz/(z*np.log(10)))**2
            + sig_lensing**2 + sig_hostcorr**2 + sig_int**2
            + 2*eta*cov_m_p2 + 2*theta*cov_m_p3 + 2*(-beta)*cov_m_c 
            + 2*eta*theta*cov_p2_p3 + 2*eta*(-beta)*cov_p2_c
            + 2*theta*(-beta)*cov_p3_c
           )
        
        return -0.5*np.sum((mu_SN - mu_cosmo)**2/D + np.log(D))

    def log_probability(params, mb, p2, p3, color, dmb, dp2, dp3, dcolor, z, dz=0.0, 
                       log_mass=0.0, dlog_mass=0.0, gamma=0.0,
                       sig_int=0.0, sig_lensing=0.0, sig_hostcorr=0.0, 
                       cov_m_p2=0.0, cov_m_p3=0.0, cov_m_c=0.0,
                       cov_p2_p3=0.0, cov_p2_c=0.0, cov_p3_c=0.0):

        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, mb, p2, p3, color, dmb, dp2, dp3, dcolor, z, dz, 
                                   log_mass, dlog_mass, gamma, sig_int, sig_lensing, sig_hostcorr, 
                                   cov_m_p2, cov_m_p3, cov_m_c, cov_p2_p3, cov_p2_c, cov_p3_c)

    ndim = len(p0)
    pos = p0 + 1e-4*np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=args)
    sampler.run_mcmc(pos, nsteps);
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    
    if plot==True:
        if labels is None:
            labels = [f'param{i}' for i in ndim]
        fig = corner.corner(samples, labels=labels)
        plt.show()
        
    results = list()
    errors = list()
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        results.append(mcmc[1])
        q = np.diff(mcmc)
        errors.append(q)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i].replace('$', ''))
        if verbose==True:
            display(Math(txt))
        
    return results, errors


def pisco_residual2(params, mb, p1, p2, p3, color, dmb, dp1, dp2, dp3, dcolor, z, dz=0.0, 
                   log_mass=0.0, dlog_mass=0.0, gamma=0.0,
                   sig_int=0.0, sig_lensing=0.0, sig_hostcorr=0.0, 
                   cov_m_p2=0.0, cov_m_p3=0.0, cov_m_c=0.0,
                   cov_p2_p3=0.0, cov_p2_c=0.0, cov_p3_c=0.0):
    
    try:
        M = params['M'].value
    except:
        M = 0.0
    
    try:
        zeta = params['zeta'].value
    except:
        zeta = 0.0
        
    try:
        eta = params['eta'].value
    except:
        eta = 0.0
    
    try:
        theta = params['theta'].value
    except:
        theta = 0.0
    
    try:
        beta = params['beta'].value
    except:
        beta = 0.0
        
    try:
        gamma = np.asarray([params['gamma'].value if mass >= 10 
                            else 0.0 for mass in log_mass])
    except:
        pass

    try:
        Omega_m = params['Omega_m'].value
        H0 = 70
        mu_cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m).distmod(z).value
    except:
        mu_cosmo = cosmo.distmod(z).value
        
    mu_SN = mb + zeta*p1  + eta*p2 + theta*p3 - beta*color  - (M + gamma)
    
    # zeta**2*(dp1/(p1*np.log(10)))**2
    D = (dmb**2 + (zeta**2)*(dp1**2) + (eta**2)*(dp2**2) + (theta**2)*(dp3**2) + (beta**2)*(dcolor**2)
            + (5*dz/(z*np.log(10)))**2
            + sig_lensing**2 + sig_hostcorr**2 + sig_int**2
            + 2*eta*cov_m_p2 + 2*theta*cov_m_p3 + 2*(-beta)*cov_m_c 
            + 2*eta*theta*cov_p2_p3 + 2*eta*(-beta)*cov_p2_c
            + 2*theta*(-beta)*cov_p3_c
           )
    
    N = len(params.keys())
    dof = len(mu_SN) - N
    chi2 = np.sum((mu_SN - mu_cosmo)**2/D)
    
    return chi2

def pisco_mcmc2(p0, args, labels=None, nwalkers=32, nsteps=1000, nburn=500, verbose=True, plot=False):

    def log_prior(params):
        
        try:
            i = labels.index('M')
            M = params[i]
            constrain_M = -20.0 < M < -17
        except:
            constrain_M = True
        try:
            i = labels.index(r'$\eta$')
            eta = params[i]
            constrain_eta = 0.0 < eta < 10.0
        except:
            constrain_eta = True
        try:
            i = labels.index(r'$\theta$')
            theta = params[i]
            constrain_theta = 0.0 < theta < 10.0
        except:
            constrain_theta = True
        try:
            i = labels.index(r'$\beta$')
            beta = params[i]
            constrain_beta = 0.0 < beta < 8.0
        except:
            constrain_beta = True
        try:
            i = labels.index(r'$\gamma$')
            gamma = params[i]
            constrain_gamma = -0.2 < gamma < 0.2
        except:
            constrain_gamma = True
        try:
            i = labels.index(r'$\Omega_m$')
            Omega_m = params[i]
            constrain_Omega_m = 0.0 < Omega_m < 1.0
        except:
            constrain_Omega_m = True
        
        if constrain_M and constrain_eta and constrain_theta and constrain_beta and constrain_gamma and constrain_Omega_m:
                return 0.0
            
        return -np.inf

    def log_likelihood(params, mb, p1, p2, p3, color, dmb, dp1, dp2, dp3, dcolor, z, dz=0.0, 
                       log_mass=0.0, dlog_mass=0.0, gamma=0.0,
                       sig_int=0.0, sig_lensing=0.0, sig_hostcorr=0.0, 
                       cov_m_p2=0.0, cov_m_p3=0.0, cov_m_c=0.0,
                       cov_p2_p3=0.0, cov_p2_c=0.0, cov_p3_c=0.0):

        
        try:
            i = labels.index('M')
            M = params[i]
        except:
            M = 0.0
        try:
            i = labels.index(r'$\zeta$')
            zeta = params[i]
        except:
            zeta = 0.0
        try:
            i = labels.index(r'$\eta$')
            eta = params[i]
        except:
            eta = 0.0
        try:
            i = labels.index(r'$\theta$')
            theta = params[i]
        except:
            theta = 0.0
        try:
            i = labels.index(r'$\beta$')
            beta = params[i]
        except:
            beta = 0.0
        try:
            i = labels.index(r'$\gamma$')
            gamma = np.asarray([params[i] if mass >= 10 else 0.0 for mass in log_mass])
        except:
            pass        
        try:
            i = labels.index(r'$\Omega_m$')
            Omega_m = params[i]
            H0 = 70
            mu_cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m).distmod(z).value
        except:
            mu_cosmo = cosmo.distmod(z).value
            
        mu_SN = mb - (M + gamma) + zeta*p1 + eta*p2 + theta*p3 - beta*color

        D = (dmb**2 + (zeta**2)*(dp1**2) + (eta**2)*(dp2**2) + (theta**2)*(dp3**2) + (beta**2)*(dcolor**2)
            + (5*dz/(z*np.log(10)))**2
            + sig_lensing**2 + sig_hostcorr**2 + sig_int**2
            + 2*eta*cov_m_p2 + 2*theta*cov_m_p3 + 2*(-beta)*cov_m_c 
            + 2*eta*theta*cov_p2_p3 + 2*eta*(-beta)*cov_p2_c
            + 2*theta*(-beta)*cov_p3_c
           )
        
        return -0.5*np.sum((mu_SN - mu_cosmo)**2/D + np.log(D))

    def log_probability(params, mb, p1, p2, p3, color, dmb, dp1, dp2, dp3, dcolor, z, dz=0.0, 
                               log_mass=0.0, dlog_mass=0.0, gamma=0.0,
                               sig_int=0.0, sig_lensing=0.0, sig_hostcorr=0.0, 
                               cov_m_p2=0.0, cov_m_p3=0.0, cov_m_c=0.0,
                       cov_p2_p3=0.0, cov_p2_c=0.0, cov_p3_c=0.0):

        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, mb, p1, p2, p3, color, dmb, dp1, dp2, dp3, dcolor, z, dz, 
                                   log_mass, dlog_mass, gamma, sig_int, sig_lensing, sig_hostcorr, 
                                   cov_m_p2, cov_m_p3, cov_m_c, cov_p2_p3, cov_p2_c, cov_p3_c)

    ndim = len(p0)
    pos = p0 + 1e-4*np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=args)
    sampler.run_mcmc(pos, nsteps);
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    
    if plot==True:
        if labels is None:
            labels = [f'param{i}' for i in ndim]
        fig = corner.corner(samples, labels=labels, truths=p0)
        plt.show()
        
    results = list()
    errors = list()
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        results.append(mcmc[1])
        q = np.diff(mcmc)
        errors.append(q)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i].replace('$', ''))
        if verbose==True:
            display(Math(txt))
        
    return results, errors
##########################################################################################

def mass_step_func(res, res_err, mass, mass_err, threshold=10):

    # divide in low and high mass samples
    maskl = np.where(mass<threshold)
    low_res = res[maskl]
    low_mass = mass[maskl]
    maskh = np.where(mass>=threshold)
    high_res = res[maskh]
    high_mass = mass[maskh]

    weights=True
    if weights:
        wl = 1/res_err[maskl]**2
        wh = 1/res_err[maskh]**2
    else:
        wl = np.ones_like(res_err[maskl])
        wh = np.ones_like(res_err[maskh])

    # low
    Nl = len(low_res)
    lowres_wmean = np.average(low_res, weights=wl)
    lowres_werr = np.sqrt(np.sum(wl*(low_res- lowres_wmean)**2)/((Nl-1)*np.sum(wl)))  # weighted standard error of the mean
    lowmass_mean = np.average(low_mass)

    # high
    Nh = len(high_res)
    highres_wmean = np.average(high_res, weights=wh)#
    highres_werr = np.sqrt(np.sum(wh*(high_res- highres_wmean)**2)/((Nh-1)*np.sum(wh)))
    highmass_mean = np.average(high_mass)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(mass, res, xerr=mass_err, yerr=res_err, fmt='o', color='k', alpha=0.2)
    ax.set_xlabel(r'log$_{10}$($M_{\rm stellar}/M_{\odot}$)')
    ax.set_ylabel(r'$\mu_{\rm SN}$ - $\mu_{\Lambda \rm CDM}$')
    ax.axvline(threshold, color='k', ls='--')
    
    ax.errorbar(lowmass_mean, lowres_wmean, yerr=lowres_werr, fmt='o', color='r', mec='k', ms=8, zorder=7)
    ax.errorbar(highmass_mean, highres_wmean, yerr=highres_werr, fmt='o', color='r', mec='k', ms=8, zorder=8)
    ax.plot([low_mass.min(), threshold], [lowres_wmean, lowres_wmean], color='r', ls='-', zorder=9)
    ax.plot([threshold, high_mass.max()], [highres_wmean, highres_wmean], color='r', ls='-', zorder=10)
    
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlim(7, 12)
    plt.grid(ls='--')

    # significance
    ms = highres_wmean - lowres_wmean
    ms_err = np.sqrt(lowres_werr**2 + highres_werr**2)
    sig = np.abs(ms/ms_err)
    print(f'Mass step = {ms:.3f} +/- {ms_err:.3f} ({sig:.1f} sigma)') 
    

def calc_hostsig(log_mass, dlog_mass, ref_sig):

    sig_host = []
    for mass, sig in zip(log_mass, dlog_mass):
        if (mass <= 10) and (mass + 3*sig > 10):
            sig_host.append(0.5 + (mass - 10)/(2*(3*sig)))

        elif (mass > 10) and (mass - 3*sig < 10):
            sig_host.append(0.5 - (10 - mass)/(2*(3*sig)))

        else:
            sig_host.append(0.0)

    sig_hostcorr = np.asarray(sig_host)*ref_sig
    return sig_hostcorr