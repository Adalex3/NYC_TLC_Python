import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import solve

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
    Vt = coo_matrix((np.ones(N), (row_indices, col_indices_t)), shape=(N, n_temporal)).toarray()

    # Vs <- as.matrix(sparseMatrix(i = 1:N, j = rep(1:n_spatial, n_temporal), x=rep(1,N)))
    col_indices_s = np.tile(np.arange(n_spatial), n_temporal)
    Vs = coo_matrix((np.ones(N), (row_indices, col_indices_s)), shape=(N, n_spatial)).toarray()

    # Sacrifice to the intercept gods
    # Vt <- Vt[,2:ncol(Vt)]
    Vt = Vt[:, 1:]
    # Vs <- Vs[,2:ncol(Vs)]
    Vs = Vs[:, 1:]

    return {'Vs': Vs, 'Vt': Vt, 'y': y}

def gp_param_bounds(Ds, Dt, kernel_func, tolerance=1e-10):
    """
    Finds reasonable upper/lower bounds on gp parameters ensuring matrices remain pd invertible
    
    :param Ds: Spatial distance matrix
    :param Dt: Temporal distance matrix
    :param kernel_func: kernel function
    :return: Minimum values for l*s, maximum values for l*t 
    """
    smin = 1.0
    Ks = kernel_func(Ds, 1.0 / smin)
    # TODO: Add try catch to solve for better error messages
    # err <- sqrt(sum(((solve(Ks) %*% Ks) - diag(1, nrow=nrow(Ks)))^2))
    err = np.sqrt(np.sum((solve(Ks, Ks) - np.eye(Ks.shape[0]))**2))
    
    while err < tolerance:
        smin = smin / 2.0
        Ks = kernel_func(Ds, 1.0 / smin)
        err = np.sqrt(np.sum((solve(Ks, Ks) - np.eye(Ks.shape[0]))**2))
    
    smin = smin * 2.0
    if smin > 0.01:
        # stop("Ds causes ill-conditioned kernel matrix, try increasing distances between spatial coordinates, e.g. Ds <- 100 * Ds")
        raise ValueError("Ds causes ill-conditioned kernel matrix, try increasing distances between spatial coordinates, e.g. Ds <- 100 * Ds")
    
    tmin = 1.0
    Kt = kernel_func(Dt, 1.0 / tmin)
    err = np.sqrt(np.sum((solve(Kt, Kt) - np.eye(Kt.shape[0]))**2))
    
    while err < tolerance:
        tmin = tmin / 2.0
        Kt = kernel_func(Dt, 1.0 / tmin)
        err = np.sqrt(np.sum((solve(Kt, Kt) - np.eye(Kt.shape[0]))**2))
    
    tmin = tmin * 2.0
    if tmin > 0.01:
        # stop("Dt casuses ill-conditioned kernel matrix, try increasing distances between temporal coordinates, e.g. Dt <- Dt * 100")
        raise ValueError("Dt casuses ill-conditioned kernel matrix, try increasing distances between temporal coordinates, e.g. Dt <- Dt * 100")
    
    ltmax = (1.0 / tmin)
    lsmax = (1.0 / smin)
    return {'ltmax': ltmax, 'lsmax': lsmax}

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
    Create the squared exponential kernel matrix e^(-dist / (ls^2))
    
    :param dist: Distance matrix
    :param ls: length scale
    """
    # return(exp(-dist / (ls^2)))
    return np.exp(-dist / (ls**2))

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
    # return(pmax(1e-6, pmin(1 - 1e-6, exp(eta) / (1 + exp(eta))))) 
    val = np.exp(eta) / (1 + np.exp(eta))
    return np.maximum(1e-6, np.minimum(1 - 1e-6, val))