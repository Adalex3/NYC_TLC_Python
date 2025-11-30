import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd
import logging

# ==============================================================================
# From setup.R
# ==============================================================================

def make_y_Vs_Vt(obs_matrix):
    """
    Create y, along with spatial and temporal design matrices from an observation matrix.
    
    :param obs_matrix: s by t matrix, where s is the number of locations, t is the number of times, each entry of the matrix is a nonnegative integer.
    :return: A Dictionary (List in R) of the following values:          
         - y: Flattened version of the observation matrix, flattened in column-major order.
         - Vs: Spatial design matrix, indicates which elements in y correspond with which positions in space
         - Vt: Temporal design matrix, indicates which elements in y correspond with which positions in time
    """
    # Assumes rows are space, columns are time
    # obs_matrix <- as.matrix(unname(obs_matrix))
    obs_matrix = np.array(obs_matrix)
    n_temporal = obs_matrix.shape[1] # ncol(obs_matrix)
    n_spatial = obs_matrix.shape[0]  # nrow(obs_matrix)
    N = n_spatial * n_temporal

    # Create y, Vs, Vt
    # y <- as.vector(obs_matrix) (Note: R flattens column-major by default)
    y = obs_matrix.flatten(order='F')
    
    # Vt <- as.matrix(sparseMatrix(i = 1:N, j = rep(1:n_temporal, each=n_spatial), x=rep(1, N)))
    # Python is 0-indexed, so we construct indices 0 to N-1
    row_indices = np.arange(N)
    col_indices_t = np.repeat(np.arange(n_temporal), n_spatial)
    Vt = coo_matrix((np.ones(N), (row_indices, col_indices_t)), shape=(N, n_temporal))

    # Vs <- as.matrix(sparseMatrix(i = 1:N, j = rep(1:n_spatial, n_temporal), x=rep(1,N)))
    col_indices_s = np.tile(np.arange(n_spatial), n_temporal)
    Vs = coo_matrix((np.ones(N), (row_indices, col_indices_s)), shape=(N, n_spatial))

    # Sacrifice to the intercept gods
    # Vt <- Vt[,2:ncol(Vt)]
    # Convert to CSR format which supports efficient slicing
    Vt = Vt.tocsr()[:, 1:]
    # Vs <- Vs[,2:ncol(Vs)]
    # Convert to CSR format which supports efficient slicing
    Vs = Vs.tocsr()[:, 1:]

    return {'Vs': Vs, 'Vt': Vt, 'y': y}

def kernel_s_combined(X, ls):
    """
    Combined ARD (Automatic Relevance Determination) kernel for spatial features.
    This is an RBF kernel where each feature has its own length scale.
    
    Assumes X is a (n_samples, n_features) array.
    'ls' is a tuple of length scales, one for each feature.
    """
    # Ensure ls is a numpy array for vectorized operations
    ls = np.asarray(ls)
    
    # Scale each feature by its length scale before computing distance
    # X_scaled has shape (n_samples, n_features)
    X_scaled = X / ls
    
    # Compute the squared Euclidean distance on the scaled features
    D2_scaled = cdist(X_scaled, X_scaled, 'sqeuclidean')
    return np.exp(-0.5 * D2_scaled)

def kernel_t_combined(X, ls):
    """
    Combined kernel for temporal features (is_weekend, hour).
    Assumes X is a 2D array where:
    - Column 0 is 'is_weekend' (categorical).
    - Column 1 is the 'hour' (continuous).
    'ls' is a tuple of length scales (ls_weekend, ls_hour).
    """
    ls_weekend, ls_hour = ls

    # RBF kernel for the 'is_weekend' feature (acts like a categorical check)
    weekends = X[:, 0].reshape(-1, 1)
    D2_weekend = cdist(weekends, weekends, 'sqeuclidean')
    K_weekend = np.exp(-0.5 * D2_weekend / ls_weekend**2)

    # RBF kernel for the 'hour' feature
    hours = X[:, 1].reshape(-1, 1)
    D2_hour = cdist(hours, hours, 'sqeuclidean')
    K_hour = np.exp(-0.5 * D2_hour / ls_hour**2)

    # Combine kernels by element-wise multiplication
    return K_hour * K_weekend

def get_gp_length_scale_bound(features, name):
    """
    Calculates a reasonable upper bound for a GP length scale parameter.

    This is a simple heuristic: a good upper bound is on the order of the
    maximum possible distance between any two points in the feature space.
    This prevents the kernel from becoming ill-conditioned by choosing a length
    scale so large that all points appear identical.

    Args:
        features (np.ndarray): The input feature matrix (n_samples, n_features).
        name (str): A name for logging purposes (e.g., 'Ds', 'Dt').

    Returns:
        float: A reasonable upper bound for the length scale.
    """
    max_dist = np.sqrt(cdist(features, features, 'sqeuclidean').max())
    logging.info(f"Determined max length scale for {name} based on max feature distance: {max_dist:.4f}")
    return max_dist

# ==============================================================================
# From svd.R
# ==============================================================================

def solve_svd(A_svd, b, threshold=1e-12):
    """
    Use SVD to solve a linear system Ax=b
    
    :param A_svd: SVD of A (Expects object with u, d, v components)
    :param b: b
    :return: x
    """
    # Get parts that have nonzero eigenvalues
    # rank <- sum(A_svd$d > threshold)
    rank = np.sum(A_svd['d'] > threshold)
    
    # Upart <- A_svd$u[,1:rank]
    Upart = A_svd['u'][:, :rank]
    # Vpart <- A_svd$v[,1:rank]
    # Note: numpy svd returns vh (V transposed). So V is vh.T
    Vpart = A_svd['v'][:, :rank] 
    # dpart <- A_svd$d[1:rank]
    dpart = A_svd['d'][:rank]
    
    # Compute solution
    # result <- Vpart %*% ((1 / dpart) * crossprod(Upart, b))
    # Note: crossprod(A, b) is t(A) %*% b
    crossprod_U_b = Upart.T @ b
    
    # Handle broadcasting for 1/dpart multiplication
    if b.ndim > 1:
        inv_dpart = (1.0 / dpart)[:, np.newaxis]
    else:
        inv_dpart = (1.0 / dpart)

    result = Vpart @ (inv_dpart * crossprod_U_b)
    return result

def mvn_sample_svd(P_svd, mu, entropy=None, threshold=1e-12):
    """
    Use SVD for precision matrix to sample from multivariate normal distribution
    
    :param P_svd: SVD of precision matrix
    :param mu: Mean of MVN to draw from
    :param entropy: Draw from MVN of correct size (can be used to draw all mvns at once for efficiency)
    """
    if entropy is None:
        # entropy <- as.matrix(rnorm(length(mu)))
        entropy = np.random.normal(size=len(mu))

    # P_half_svd <- list(u=P_svd$u, d=sqrt(P_svd$d), v=P_svd$u)
    # Note: P is symmetric precision matrix, so u=v. 
    P_half_svd = {
        'u': P_svd['u'],
        'd': np.sqrt(P_svd['d']),
        'v': P_svd['u'] # Using u for v as per source
    }
    
    # varPart <- solve_svd(P_half_svd, entropy, threshold = sqrt(threshold))
    varPart = solve_svd(P_half_svd, entropy, threshold=np.sqrt(threshold))
    return mu + varPart

# ==============================================================================
# From ZINB_GP.R (Helper functions)
# ==============================================================================

def noise_mix(A, noise_ratio):
    """
    Create the following matrix: ratio * A + (1-ratio) * I
    
    :param A: Matrix to normalize (square)
    :param noise_ratio: Noise mixing ratio
    """
    # return(noise_ratio * A + diag(1 - noise_ratio, nrow=nrow(A)))
    return noise_ratio * A + np.diag(np.full(A.shape[0], 1 - noise_ratio))

def kernel(dist, ls):
    """
    Create the squared exponential kernel matrix e^(-dist / ls).
    Assumes dist is already squared.
    
    :param dist: Distance matrix
    :param ls: length scale
    """
    return np.exp(-dist / ls)

def nullcheck(value, default):
    """
    Returns default if value is null
    
    :param value: Nullable
    :param default: Default value
    """
    if value is None:
        return default
    return value

# ==============================================================================
# From estimation.R
# ==============================================================================

def sigmoid(eta):
    """
    Compute sigmoid function, clip properly to prevent infinity/nan
    """
    # eta <- pmin(700, eta)
    eta = np.minimum(700, eta)
    
    # Numerically stable implementation of sigmoid
    # Handles large positive and negative values of eta without overflow.
    val = np.where(
        eta >= 0,
        1 / (1 + np.exp(-eta)),
        np.exp(eta) / (1 + np.exp(eta))
    )
    return np.maximum(1e-6, np.minimum(1 - 1e-6, val))