import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from sklearn.utils.extmath import randomized_svd
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy.stats import gamma, beta, multivariate_normal, invgamma, uniform, truncnorm, nbinom, norm
from scipy.linalg import block_diag
from tqdm import tqdm
import logging

# Import helper functions from our utils module
from src.model_module.utils import (
    noise_mix, nullcheck, sigmoid, solve_svd, mvn_sample_svd,
    kernel_s_combined, kernel_t_combined
)
from scipy.sparse import hstack
from scipy.sparse import diags

# ==============================================================================
# Statistical Wrappers (to match R function signatures)
# ==============================================================================

def rtnorm(n, mean, sd, lower, upper):
    """Wrapper for truncated normal sampling matching R's msm::rtnorm"""
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=n)

def dtnorm(x, mean, sd, lower, upper, log=False):
    """Wrapper for truncated normal density matching R's msm::dtnorm"""
    # Ensure inputs are numpy arrays to handle both scalar and vector/tuple inputs for ARD kernels
    x = np.asarray(x)
    mean = np.asarray(mean)
    sd = np.asarray(sd)
    
    a, b = (lower - mean) / sd, (upper - mean) / sd
    if log:
        return truncnorm.logpdf(x, a, b, loc=mean, scale=sd)
    return truncnorm.pdf(x, a, b, loc=mean, scale=sd)

def dgamma(x, shape, rate, log=False):
    """Wrapper for Gamma density matching R's stats::dgamma (using rate = 1/scale)"""
    if log:
        return gamma.logpdf(x, a=shape, scale=1.0/rate)
    return gamma.pdf(x, a=shape, scale=1.0/rate)

def dbeta(x, shape1, shape2, log=False):
    """Wrapper for Beta density matching R's stats::dbeta"""
    if log:
        return beta.logpdf(x, shape1, shape2)
    return beta.pdf(x, shape1, shape2)

def dmvnorm(x, mean, sigma, log=False, chol_sigma=None):
    """
    Wrapper for MVN density matching R's mvtnorm::dmvnorm.
    Optimized to use a pre-computed Cholesky factor of the covariance matrix if provided.
    """
    # If Cholesky factor is not provided, fall back to the standard SciPy implementation.
    if chol_sigma is None:
        if log:
            return multivariate_normal.logpdf(x, mean=mean, cov=sigma)
        return multivariate_normal.pdf(x, mean=mean, cov=sigma)

    # Optimized path using Cholesky decomposition L, where sigma = L @ L.T
    L = chol_sigma
    k = len(x)
    dev = x - mean
    
    # Solve L @ v = dev, then v = L_inv @ dev
    v = np.linalg.solve(L, dev)
    log_det_sigma = 2 * np.sum(np.log(np.diag(L)))
    log_pdf = -0.5 * (k * np.log(2 * np.pi) + log_det_sigma + v.T @ v)

    if log:
        return log_pdf
    return np.exp(log_pdf)

def rinvgamma(n, shape, scale):
    """Wrapper for Inverse Gamma sampling matching LaplacesDemon::rinvgamma"""
    return invgamma.rvs(a=shape, scale=scale, size=n)

def rpg(n, h, z):
    """
    Wrapper for Polya-Gamma sampling matching BayesLogit::rpg.
    Enforces float64 types to prevent C++ buffer mismatch errors.
    """
    try:
        from pypolyagamma import PyPolyaGamma
        pg = PyPolyaGamma()
        
        # FIX: Force strict float64 (double) type for C++ compatibility
        if np.isscalar(h):
            h_vec = np.full(n, h, dtype=np.float64)
        else:
            h_vec = np.array(h, dtype=np.float64)
            
        z_vec = np.array(z, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        
        pg.pgdrawv(h_vec, z_vec, out)
        return out
    except ImportError:
        # Fallback if library is missing (for testing without C compiler)
        return np.ones(n, dtype=np.float64)

# ==============================================================================
# From ZINB_GP.R
# ==============================================================================

def update_ls_sigma_noise(ls, sigma, noise_ratio, gpdraw, K, features, lsPrior, sigmaPrior, noisePrior, kernel_func, param_name="param"):
    """
    Update kernel parameters for a GP
    
    :param ls: Current length scale
    :param sigma: Current sigma
    :param noise_ratio: Current noise ratio
    :param gpdraw: Last draw from the gp with these parameters
    :param K: Current kernel matrix
    :param features: Feature matrix for the kernel
    :param lsPrior: prior information for length scale, needs mh_sd, max, a, b
    :param sigmaPrior: prior information for sigma, needs a, b
    :param noisePrior: prior information for noise_ratio, needs mh_sd, a, b
    :param param_name: A name for the parameter set for logging purposes (e.g., "Spatial Logistic")
    :return: A Dictionary of the following sampled values:
         - ls: Length scale
         - sigma: sigma
         - noise_ratio: noise ratio
         - K: Kernel matrix
         - K_inv: Inverse of kernel matrix
    """
    logger = logging.getLogger(__name__)

    # update ls
    # Handle single or multiple length scales (for ARD kernels)
    is_ard = isinstance(ls, (list, tuple, np.ndarray))
    if is_ard:
        ls_array = np.array(ls)
        proposal_array = rtnorm(len(ls_array), mean=ls_array, sd=lsPrior['mh_sd'], lower=1e-6, upper=lsPrior['max'])
        proposal = tuple(proposal_array)
    else:
        proposal = rtnorm(1, mean=ls, sd=lsPrior['mh_sd'], lower=1e-6, upper=lsPrior['max'])[0]
    
    logger.debug(f"Updating ls for {param_name}. Current ls: {ls}")
    logger.debug(f"  - ls proposal: {proposal}")
    
    accepted_ls = False
    K_star = sigma**2 * noise_mix(kernel_func(features, proposal), noise_ratio)
    
    zero_mean = np.zeros(len(gpdraw))
    try:
        # --- Optimization ---
        # Compute the Cholesky factor of the current kernel matrix K once.
        # This avoids re-computing it in both the ls and noise_ratio update steps.
        L = np.linalg.cholesky(K)

        # --- Optimization ---
        # Pre-compute the Cholesky decomposition of the proposal kernel matrix.
        # This avoids a costly O(n^3) decomposition inside dmvnorm.
        L_star = np.linalg.cholesky(K_star)
        # Pass the pre-computed Cholesky factors to dmvnorm.
        likelihood_ls = dmvnorm(gpdraw, mean=zero_mean, sigma=K_star, log=True, chol_sigma=L_star) - \
                        dmvnorm(gpdraw, mean=zero_mean, sigma=K, log=True, chol_sigma=L)

        prior_ls = np.sum(dgamma(proposal, shape=lsPrior['a'], rate=lsPrior['b'], log=True)) - \
                   np.sum(dgamma(ls, shape=lsPrior['a'], rate=lsPrior['b'], log=True))
        
        trans_ls = np.sum(dtnorm(ls, mean=proposal, sd=lsPrior['mh_sd'], lower=1e-6, upper=lsPrior['max'], log=True)) - \
                   np.sum(dtnorm(proposal, mean=ls, sd=lsPrior['mh_sd'], lower=1e-6, upper=lsPrior['max'], log=True))

        posterior_ls = likelihood_ls + prior_ls + trans_ls

        if not np.isnan(posterior_ls) and np.log(uniform.rvs()) < posterior_ls:
            ls = proposal
            K = K_star
            accepted_ls = True
    except np.linalg.LinAlgError:
        logger.debug("  - SVD did not converge for ls proposal. Proposal rejected.")

    logger.debug(f"  - ls proposal accepted: {accepted_ls}")

    # Update noise ratio
    logger.debug(f"Updating noise_ratio for {param_name}. Current noise_ratio: {noise_ratio:.4f}")
    eps_nr = 0.005
    proposal = rtnorm(1, mean=noise_ratio, sd=noisePrior['mh_sd'], lower=eps_nr, upper=1 - eps_nr)[0]
    logger.debug(f"  - noise_ratio proposal: {proposal:.4f}")

    accepted_nr = False
    K_star = sigma**2 * noise_mix(kernel_func(features, ls), proposal)
    
    try:
        # --- Optimization ---
        # Reuse the pre-computed Cholesky factor 'L' for the current kernel K.
        L_star = np.linalg.cholesky(K_star)
        likelihood_nr = dmvnorm(gpdraw, mean=zero_mean, sigma=K_star, log=True, chol_sigma=L_star) - \
                        dmvnorm(gpdraw, mean=zero_mean, sigma=K, log=True, chol_sigma=L)

        prior_nr = dbeta(proposal, shape1=noisePrior['a'], shape2=noisePrior['b'], log=True) - \
                   dbeta(noise_ratio, shape1=noisePrior['a'], shape2=noisePrior['b'], log=True)

        trans_ls = dtnorm(noise_ratio, mean=proposal, sd=noisePrior['mh_sd'], lower=eps_nr, upper=1 - eps_nr, log=True) - \
                   dtnorm(proposal, mean=noise_ratio, sd=noisePrior['mh_sd'], lower=eps_nr, upper=1 - eps_nr, log=True)

        posterior_nr = likelihood_nr + prior_nr + trans_ls

        if not np.isnan(posterior_nr) and np.log(uniform.rvs()) < posterior_nr:
            noise_ratio = proposal
            accepted_nr = True
    except np.linalg.LinAlgError:
        logger.debug("  - SVD did not converge for noise_ratio proposal. Proposal rejected.")
        
    logger.debug(f"  - noise_ratio proposal accepted: {accepted_nr}")

    # update sigma
    logger.debug(f"Updating sigma for {param_name}. Current sigma: {sigma:.4f}")
    K_nosigma = noise_mix(kernel_func(features, ls), noise_ratio)
    
    try:
        K_nosigma_inv = np.linalg.solve(K_nosigma, np.eye(K_nosigma.shape[0]))
        K_nosigma_inv = (K_nosigma_inv + K_nosigma_inv.T) / 2

        a_new = sigmaPrior['a'] + 0.5 * len(gpdraw)
        b_new = sigmaPrior['b'] + 0.5 * (gpdraw.T @ K_nosigma_inv @ gpdraw)
        
        scale_val = b_new if np.isscalar(b_new) else b_new.item()
        sigma_sq = rinvgamma(n=1, shape=a_new, scale=scale_val)[0]
        sigma = np.sqrt(sigma_sq)
        logger.debug(f"  - New sigma sampled: {sigma:.4f}")

        K = K_nosigma * sigma_sq
        K_inv = K_nosigma_inv * (1.0 / sigma_sq)
    except np.linalg.LinAlgError:
        logger.debug("  - Could not solve for K_nosigma_inv. Sigma update skipped.")


    return {'ls': ls, 'sigma': sigma, 'noise_ratio': noise_ratio, 'K': K, 'K_inv': K_inv}


def ZINB_GP(X, y, coords, Vs, Vt, Ds_features, Dt_features, priors, nsim, burn, thin=1, save_ypred=False, 
            print_iter=100, print_progress=False, ltPrior=None, lsPrior=None, 
            sigmaPrior=None, noisePrior=None, mh_sd_r=None, **kwargs):
    """
    Run the ZINB NNGP model.
    Includes numerical stability fixes and strict type casting for Python.
    """
    logger = logging.getLogger(__name__)

    # X is the design matrix with dimension N*p
    # x is the vector with length N
    # y is the count response with length N
    n_locs = Ds_features.shape[0]
    n_times = Dt_features.shape[0]
    n = n_locs - 1 # number of spatial random effects
    N = X.shape[0] # number of observations
    p = X.shape[1] # dimension of alpha and beta
    n_time_points = n_times - 1 # number of temporal random effects
    logger.info(f"Initializing ZINB-GP model with N={N}, p={p}, n_locs={n_locs}, n_time={n_times}")

    print("here1")

    # Get prior bounds from the input dictionary
    lsmax = priors.get('lsPrior', {}).get('max', 10.0)
    print("here2")
    ltmax = priors.get('ltPrior', {}).get('max', 10.0)
    print("here3")
    logger.debug(f"Using GP length scale bounds from priors: lsmax={lsmax:.4f}, ltmax={ltmax:.4f}")

    ##########
    # Priors #
    ##########

    ####### priors for alpha and beta ######
    T0a = np.eye(p) * 100
    T0b = np.eye(p) * 100
    sd_r = nullcheck(mh_sd_r, 0.4)

    print("here4")
    
    ####### kernel hyperparameters  ######
    ltPrior = nullcheck(ltPrior, {'max': ltmax, 'mh_sd': 0.5, 'a': 1, 'b': 0.001})
    lsPrior = nullcheck(lsPrior, {'max': lsmax, 'mh_sd': 0.5, 'a': 1, 'b': 0.001})
    sigmaPrior = nullcheck(sigmaPrior, {'a': 0.01, 'b': 0.1})
    noisePrior = nullcheck(noisePrior, {'a': 1.5, 'b': 1.5, 'mh_sd': 0.2})
    logger.debug("Priors have been set.")

    # Model init
    r = 1.0 # Ensure r is float
    y1 = np.zeros(N) # At risk indicator (this is W in paper)
    y1[y > 0] = 1 # If y>0, then at risk w.p. 1

    #########
    # Inits #
    #########

    #################
    # Fixed Effects #
    #################
    y_ind = np.zeros(N) 
    y_ind[y != 0] = 1

    # If there are fixed effects, get initial values from GLMs.
    if p > 0:
        m1 = sm.GLM(y_ind, X, family=Binomial()).fit()
        alpha1 = m1.params
        logger.debug(f"Initial alpha from GLM (Binomial): {alpha1}")

        mask_nz = (y != 0)
        if np.sum(mask_nz) > 0:
            try:
                # Try to fit a Negative Binomial GLM for a good starting point.
                # This can be numerically unstable if the data subset is difficult.
                m2 = NegativeBinomial(y[mask_nz], X[mask_nz], loglike_method='nb2').fit(disp=0, maxiter=100)
                beta = m2.params[:-1] # Exclude the dispersion parameter alpha from statsmodels
                if np.any(np.isnan(beta)):
                    logger.warning("GLM for initial beta resulted in NaNs. Defaulting to zeros.")
                    beta = np.zeros(p)
            except Exception as e:
                logger.warning(f"Could not fit initial GLM for beta, defaulting to zeros. Error: {e}")
                beta = np.zeros(p)
        else:
            beta = np.zeros(p)
    else:
        # If p=0 (no fixed effects), initialize alpha and beta as empty arrays.
        alpha1 = np.array([])
        beta = np.zeros(p)

    logger.debug(f"Initial beta: {beta}")
    
    # The terms X @ alpha1 and X @ beta will correctly be zero if p=0
    eta1 = (X @ alpha1) if p > 0 else np.zeros(N)
    eta2 = (X @ beta) if p > 0 else np.zeros(N)
    p_at_risk = sigmoid(eta1) 

    q = 1 / (1 + np.exp(eta2))
    theta = p_at_risk * (q**r) / (p_at_risk * (q**r) + 1 - p_at_risk) 
    
    mask_z = (y == 0)
    y1[mask_z] = np.random.binomial(1, theta[mask_z])

    if p > 0:
        m1 = sm.GLM(y1, X, family=Binomial()).fit()
        alpha = m1.params
    else:
        alpha = np.array([])

    logger.debug(f"Refined initial alpha after sampling y1: {alpha}")

    noise_ratio_t1 = 0.5
    noise_ratio_t2 = 0.5
    noise_ratio_s1 = 0.5
    noise_ratio_s2 = 0.5

    ##########################
    # Spatial Random Effects #
    ##########################
    # Initialize length scales as tuples for ARD kernels
    l1s = (1.0, 1.0)
    l2s = (1.0, 1.0)
    sigma1s = sigma2s = 2.0
    # We only need kernels for the n random effects (n x n matrix)
    Ds_features_re = Ds_features[1:, :]
    Ks_bin = sigma1s**2 * noise_mix(kernel_s_combined(Ds_features_re, l1s), noise_ratio_s1)
    Ks_bin_inv = np.linalg.solve(Ks_bin, np.eye(Ks_bin.shape[0])) 
    Ks_bin_inv = (Ks_bin_inv + Ks_bin_inv.T) / 2

    print("helloooo")

    Ks_nb = sigma2s**2 * noise_mix(kernel_s_combined(Ds_features_re, l2s), noise_ratio_s2)
    print("helloooo2")
    Ks_nb_inv = np.linalg.solve(Ks_nb, np.eye(Ks_nb.shape[0])) # time sink (~15sec)
    print("helloooo3")
    Ks_nb_inv = (Ks_nb_inv + Ks_nb_inv.T) / 2
    print("helloooo4")

    # --- Optimized Multivariate Normal Sampling ---
    # The original `multivariate_normal.rvs(cov=K)` is slow and memory-heavy for large
    # covariance matrices because it may perform a costly decomposition internally.
    # A much faster method is to compute the Cholesky decomposition of the covariance
    # matrix (K = L @ L.T) and then generate the sample as `a = L @ z`, where `z` is a
    # vector of standard normal random variables.

    # 1. Perform Cholesky decomposition on the covariance matrices.
    L_bin = np.linalg.cholesky(Ks_bin)
    L_nb = np.linalg.cholesky(Ks_nb)

    # 2. Generate samples using the Cholesky factors.
    a = L_bin @ np.random.normal(size=n)
    print("helloooo5")
    c = L_nb @ np.random.normal(size=n)
    print("helloooo6")
    logger.debug(f"Initialized spatial random effects 'a' (shape: {a.shape}) and 'c' (shape: {c.shape}) for {n} locations")

    #################
    # Temporal Random Effects #
    #################
    sigma1t = sigma2t = 2.0
    l1t = (1.0, 1.0)
    l2t = (1.0, 1.0)
    Dt_features_re = Dt_features[1:, :]
    Kt_bin = sigma1t**2 * noise_mix(kernel_t_combined(Dt_features_re, l1t), noise_ratio_t1)
    Kt_bin_inv = np.linalg.solve(Kt_bin, np.eye(Kt_bin.shape[0]))
    Kt_bin_inv = (Kt_bin_inv + Kt_bin_inv.T) / 2

    Kt_nb = sigma2t**2 * noise_mix(kernel_t_combined(Dt_features_re, l2t), noise_ratio_t2)
    Kt_nb_inv = np.linalg.solve(Kt_nb, np.eye(Kt_nb.shape[0]))
    Kt_nb_inv = (Kt_nb_inv + Kt_nb_inv.T) / 2
    
    b = multivariate_normal.rvs(cov=Kt_bin)
    d = multivariate_normal.rvs(cov=Kt_nb)
    logger.debug(f"Initialized temporal random effects 'b' (shape: {b.shape}) and 'd' (shape: {d.shape}) for {n_time_points} time points")

    ############
    # Num Sims #
    ############
    lastit = int((nsim - burn) / thin) 

    #########
    # Store #
    #########
    Beta = np.zeros((lastit, p))
    Alpha = np.zeros((lastit, p))
    R_store = np.zeros(lastit) 
    A = np.zeros((lastit, n))
    C = np.zeros((lastit, n))
    B = np.zeros((lastit, n_time_points))
    D_store = np.zeros((lastit, n_time_points)) 
    
    L1t = np.zeros((lastit, 2))
    Sigma1t = np.zeros(lastit)
    Noise1t = np.zeros(lastit)
    L2t = np.zeros((lastit, 2))
    Sigma2t = np.zeros(lastit)
    Noise2t = np.zeros(lastit)
    L1s = np.zeros((lastit, 2))
    Sigma1s = np.zeros(lastit)
    Noise1s = np.zeros(lastit)
    L2s = np.zeros((lastit, 2))
    Sigma2s = np.zeros(lastit)
    Noise2s = np.zeros(lastit)
    
    Y_pred = None
    y1s = None
    if save_ypred:
        Y_pred = np.full((lastit, N), np.nan)
        y1s = np.full((lastit, N), np.nan)


    ########
    # MCMC #
    ########
    # Use sparse hstack because Vs and Vt are now sparse matrices.
    # We convert X to a sparse format (CSR is efficient for row slicing) before stacking.
    from scipy.sparse import csr_matrix
    XV = hstack([csr_matrix(X), Vs, Vt])
    
    # Create the loop iterator
    iterations = range(1, nsim + 1)
    
    # Wrap with tqdm for a progress bar if print_progress is True
    if print_progress:
        iterations = tqdm(iterations, desc="MCMC Sampling")
        # This inner progress bar will show steps within each iteration
        inner_bar = tqdm(total=7, desc="Iter Steps", leave=False, bar_format='{desc}: {percentage:3.0f}%|{bar}|')

    for i in iterations:
        if print_progress:
            inner_bar.reset()
            inner_bar.set_description(f"Iter {i} Steps")

        # Update priors
        Sigma0_bin_inv = block_diag(Ks_bin_inv, Kt_bin_inv)
        Sigma0_nb_inv = block_diag(Ks_nb_inv, Kt_nb_inv)
        T0_bin = block_diag(T0a, Sigma0_bin_inv)
        T0_nb = block_diag(T0b, Sigma0_nb_inv)
        logger.debug(f"Iter {i}: Priors updated.")

        # -----------------------------------------------------
        # Update alpha, a, b (Logistic Component)
        # -----------------------------------------------------
        if p > 0:
            mu = X @ alpha + Vs @ a + Vt @ b
        else:
            mu = Vs @ a + Vt @ b
        w = rpg(N, 1, mu) 
        
        logger.debug(f"Iter {i}: Logistic - mu shape: {mu.shape}, w shape: {w.shape}")
        # NOTE: We skip calculating 'z' directly to avoid divide-by-zero errors when w is small.
        # Instead, we use the algebraic identity: X' W z = X' (y - 1/2)
        # Old (unstable): z = (y1 - 1 / 2) / w
        
        # To perform row-wise scaling on the sparse matrix XV, we can't use broadcasting
        # with a column vector. Instead, we create a sparse diagonal matrix from the
        # weights and use matrix multiplication. diag(sqrt(w)) @ XV scales each row i
        # of XV by sqrt(w)[i].
        sqrt_w_XV = diags(np.sqrt(w)) @ XV
        crossprod_val = sqrt_w_XV.T @ sqrt_w_XV

        # --- Optimized SVD Calculation ---
        # The full SVD (np.linalg.svd) is computationally expensive. For large matrices
        # inside an MCMC loop, this becomes a major bottleneck.
        # We can use a randomized SVD, which computes an approximate low-rank decomposition
        # much more quickly. This is highly effective as the precision matrices often have
        # a fast-decaying singular value spectrum.
        # The number of components is a tunable parameter; a value around 150 is often a
        # good starting point for matrices of this size (~200x200) to balance speed and accuracy.
        svd_vinv_u, svd_vinv_s, svd_vinv_v = randomized_svd(crossprod_val + T0_bin, n_components=150, random_state=None)
        svd_vinv = {'u': svd_vinv_u, 'd': svd_vinv_s, 'v': svd_vinv_v.T}

        # Stable calculation for RHS: XV.T @ (y1 - 0.5)
        rhs = XV.T @ (y1 - 0.5)
        m = solve_svd(svd_vinv, rhs)
        
        alphaab = mvn_sample_svd(svd_vinv, m)
        alpha = alphaab[0:p]
        a = alphaab[p:(p + n)]
        b = alphaab[(p + n):]
        logger.debug(f"Iter {i}: Logistic - Updated alpha, a, b.")
        if print_progress: inner_bar.update(1)

        # -----------------------------------------------------
        # Update at-risk indicator y1
        # -----------------------------------------------------
        logger.debug(f"Iter {i}: Updating at-risk indicator y1.")
        if p > 0:
            eta1 = X @ alpha + Vs @ a + Vt @ b
            eta2 = X @ beta + Vs @ c + Vt @ d
        else:
            eta1 = Vs @ a + Vt @ b
            eta2 = Vs @ c + Vt @ d
        pi_val = sigmoid(eta1) 
        q = 1 / (1 + np.exp(eta2)) 
        theta = pi_val * (q**r) / (pi_val * (q**r) + 1 - pi_val) 
        
        mask_z = (y == 0)
        num_to_update = np.sum(mask_z)
        y1[mask_z] = np.random.binomial(1, theta[mask_z])
        logger.debug(f"Iter {i}: y1 updated for {num_to_update} zero-count observations.")
        
        # FIX: Ensure N1 is an integer for rpg()
        N1 = int(np.sum(y1))
        if print_progress: inner_bar.update(1)

        # -----------------------------------------------------
        # Update r
        # -----------------------------------------------------
        logger.debug(f"Iter {i}: Updating dispersion parameter r. Current r: {r:.4f}")
        rnew = rtnorm(1, r, sd_r, lower=0, upper=np.inf)[0]
        mask_y1 = (y1 == 1)
        
        # Check if we have any at-risk population
        if np.sum(mask_y1) > 0:
            ll_new = np.sum(nbinom.logpmf(y[mask_y1], rnew, 1 - q[mask_y1]))
            ll_old = np.sum(nbinom.logpmf(y[mask_y1], r, 1 - q[mask_y1]))
            
            ratio = ll_new - ll_old + \
                dtnorm(r, rnew, sd_r, 0, np.inf, log=True) - \
                dtnorm(rnew, r, sd_r, 0, np.inf, log=True)
            
            if np.log(uniform.rvs()) < ratio:
                r = rnew
                logger.debug(f"Iter {i}: Dispersion parameter r updated to {r:.4f}")
        if print_progress: inner_bar.update(1)

        # -----------------------------------------------------
        # Update Hyperparams (Logistic)
        # -----------------------------------------------------
        logger.debug(f"Iter {i}: Updating logistic component hyperparameters.")
        out = update_ls_sigma_noise(l1t, sigma1t, noise_ratio_t1, b, Kt_bin, Dt_features_re, ltPrior, sigmaPrior, noisePrior, kernel_t_combined, "Logistic Temporal")
        l1t, sigma1t, noise_ratio_t1 = out['ls'], out['sigma'], out['noise_ratio']
        Kt_bin, Kt_bin_inv = out['K'], out['K_inv']
        logger.debug(f"Iter {i}: Logistic Temporal - l1t={l1t}, sigma1t={sigma1t:.4f}, noise1t={noise_ratio_t1:.4f}")

        out = update_ls_sigma_noise(l1s, sigma1s, noise_ratio_s1, a, Ks_bin, Ds_features_re, lsPrior, sigmaPrior, noisePrior, kernel_s_combined, "Logistic Spatial")
        l1s, sigma1s, noise_ratio_s1 = out['ls'], out['sigma'], out['noise_ratio']
        Ks_bin, Ks_bin_inv = out['K'], out['K_inv']
        logger.debug(f"Iter {i}: Logistic Spatial - l1s={l1s}, sigma1s={sigma1s:.4f}, noise1s={noise_ratio_s1:.4f}")
        if print_progress: inner_bar.update(1)

        # -----------------------------------------------------
        # Update beta, c, d (Count Component)
        # -----------------------------------------------------
        logger.debug(f"Iter {i}: Updating count component parameters (beta, c, d). N1={N1}")
        if p > 0:
            eta = X[mask_y1] @ beta + Vs[mask_y1] @ c + Vt[mask_y1] @ d
        else:
            eta = Vs[mask_y1] @ c + Vt[mask_y1] @ d
        w = rpg(N1, y[mask_y1] + r, eta)
        
        logger.debug(f"Iter {i}: Count - eta shape: {eta.shape}, w shape: {w.shape}")
        # Stable calculation: X' W z = X' (y - r)/2
        # Old (unstable): z = (y[mask_y1] - r) / (2 * w)

        # Perform row-wise scaling on the sparse matrix subset using a diagonal matrix.
        # This is the sparse-matrix equivalent of `np.sqrt(w)[:, None] * dense_matrix`.
        sqrt_w_XV_sub = diags(np.sqrt(w)) @ XV[mask_y1]
        crossprod_val = sqrt_w_XV_sub.T @ sqrt_w_XV_sub

        # --- Optimized SVD Calculation ---
        # As with the logistic component, we use randomized SVD here to accelerate the
        # decomposition of the precision matrix for the count component parameters.
        # This provides a significant performance boost within the MCMC loop.
        svd_vinv_u, svd_vinv_s, svd_vinv_v = randomized_svd(crossprod_val + T0_nb, n_components=150, random_state=None)
        svd_vinv = {'u': svd_vinv_u, 'd': svd_vinv_s, 'v': svd_vinv_v.T}

        # Stable calculation for RHS
        rhs = XV[mask_y1].T @ (0.5 * (y[mask_y1] - r))
        m = solve_svd(svd_vinv, rhs)
        
        betacd = mvn_sample_svd(svd_vinv, m)
        beta = betacd[0:p]
        c = betacd[p:(p + n)]
        d = betacd[(p + n):]
        logger.debug(f"Iter {i}: Count - Updated beta, c, d.")
        if print_progress: inner_bar.update(1)

        # -----------------------------------------------------
        # Update Hyperparams (Count)
        # -----------------------------------------------------
        logger.debug(f"Iter {i}: Updating count component hyperparameters.")
        out = update_ls_sigma_noise(l2t, sigma2t, noise_ratio_t2, d, Kt_nb, Dt_features_re, ltPrior, sigmaPrior, noisePrior, kernel_t_combined, "Count Temporal")
        l2t, sigma2t, noise_ratio_t2 = out['ls'], out['sigma'], out['noise_ratio']
        Kt_nb, Kt_nb_inv = out['K'], out['K_inv']
        logger.debug(f"Iter {i}: Count Temporal - l2t={l2t}, sigma2t={sigma2t:.4f}, noise2t={noise_ratio_t2:.4f}")

        out = update_ls_sigma_noise(l2s, sigma2s, noise_ratio_s2, c, Ks_nb, Ds_features_re, lsPrior, sigmaPrior, noisePrior, kernel_s_combined, "Count Spatial")
        l2s, sigma2s, noise_ratio_s2 = out['ls'], out['sigma'], out['noise_ratio']
        Ks_nb, Ks_nb_inv = out['K'], out['K_inv']
        logger.debug(f"Iter {i}: Count Spatial - l2s={l2s}, sigma2s={sigma2s:.4f}, noise2s={noise_ratio_s2:.4f}")
        if print_progress: inner_bar.update(1)

        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------
        if (i > burn) and ((i - burn) % thin == 0):
            logger.debug(f"Iter {i}: Storing posterior samples.")
            j = int((i - burn) / thin) - 1 
            
            Alpha[j, :] = alpha
            Beta[j, :] = beta 
            A[j, :] = a
            B[j, :] = b
            C[j, :] = c
            D_store[j, :] = d # random effects
            
            L1t[j, :] = l1t
            Noise1t[j] = noise_ratio_t1
            Sigma1t[j] = sigma1t # temporal hyperparameters
            
            L2t[j, :] = l2t
            Noise2t[j] = noise_ratio_t2
            Sigma2t[j] = sigma2t # temporal hyperparameters
            
            L1s[j, :] = l1s
            Noise1s[j] = noise_ratio_s1
            Sigma1s[j] = sigma1s # spatial hyperparameters
            
            L2s[j, :] = l2s
            Noise2s[j] = noise_ratio_s2
            Sigma2s[j] = sigma2s # spatial hyperparameters
            
            R_store[j] = r
            if save_ypred:
                Y_pred[j, :] = estimate(X, alpha, beta, Vs, Vt, a, b, c, d, r)
                y1s[j, :] = y1
        if print_progress:
            inner_bar.update(1) # Final update for storing step

    if print_progress:
        inner_bar.close()

    # Put the results into a list
    results = {
        'Alpha': Alpha, 'Beta': Beta, 'A': A, 'B': B, 'C': C, 'D': D_store,
        'L1t': L1t, 'Sigma1t': Sigma1t, 'Noise1t': Noise1t, 
        'L2t': L2t, 'Sigma2t': Sigma2t, 'Noise2t': Noise2t,
        'L1s': L1s, 'Sigma1s': Sigma1s, 'Noise1s': Noise1s, 
        'L2s': L2s, 'Sigma2s': Sigma2s, 'Noise2s': Noise2s,
        'R': R_store
    }
    if save_ypred:
        results['Y_pred'] = Y_pred
        results['at_risk'] = y1s
        
    return results

# ==============================================================================
# From estimation.R
# ==============================================================================

def estimate(X, alpha, beta, Vs, Vt, a, b, c, d, r):
    """Draw y from posterior distribution"""
    N = X.shape[0]
    
    # Calculate etas
    eta1 = X @ alpha + Vs @ a + Vt @ b
    p_at_risk = sigmoid(eta1) # at-risk probability
    u = np.random.binomial(1, p_at_risk) # at-risk indicator
    
    # Python handling of indexing with boolean array u
    mask_u = (u == 1)
    
    # Linear predictor for count part
    # Note: Using @ for matrix mult. X[mask_u] is a matrix.
    eta2 = X[mask_u] @ beta + Vs[mask_u] @ c + Vt[mask_u] @ d 

    psi = sigmoid(eta2) # Prob of success
    mu = r * psi / (1 - psi) # NB mean
    y = np.zeros(N) # Response
    
    # Draw from posterior
    # R: rnbinom(n, size=r, prob=(1-psi))
    # Python: nbinom.rvs(n=r, p=(1-psi))
    y[mask_u] = nbinom.rvs(n=r, p=(1 - psi))
    
    return y

def predict(X, Ds_new, Dt_new, Vs_new, Vt_new, output):
    """
    Predict at new locations/times, provides predictions along with estimated variance.
    (Source function appeared incomplete, transcribed as is)
    """
    l1t = output['L1t']
    l2t = output['L2t']
    phi_bin = output['Phi_bin']
    phi_nb = output['Phi_nb']
    # Code ends abruptly in source