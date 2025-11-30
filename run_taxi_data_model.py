
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import logging

from src.model_module.core import ZINB_GP
from src.model_module.utils import make_y_Vs_Vt, get_gp_length_scale_bound, kernel_s_combined, kernel_t_combined

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_model_with_taxi_data(csv_path: str):
    """
    Loads and processes NYC taxi data to run the ZINB-GP model.

    Args:
        csv_path (str): The full path to the taxi data CSV file.
    """
    # --- 1. Load and Prepare Data ---
    logging.info(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {csv_path}. Please update the path.")
        return

    # For consistency, sort the dataframe early
    df = df.sort_values(by=['grid_x', 'grid_y', 'is_weekend', 'hour']).reset_index(drop=True)

    # --- 2. Create Observation Matrix and Coordinate Inputs ---
    logging.info("Pivoting data to create observation matrix...")
    
    # Pivot to get (locations x time) matrix for ride counts
    # Fill missing values with 0, assuming no record means no rides
    obs_matrix = df.pivot_table(
        index=['grid_y', 'grid_x'],
        columns=['is_weekend', 'hour'],
        values='ride_count'
    ).fillna(0)
    
    # Get the unique coordinates corresponding to the location_id index
    coords = np.array(obs_matrix.index.to_list())

    # --- 3. Generate y, Vs, and Vt using the helper function ---
    logging.info("Generating y, Vs, and Vt from observation matrix...")
    data_struct = make_y_Vs_Vt(obs_matrix.values)
    y, Vs, Vt = data_struct['y'], data_struct['Vs'], data_struct['Vt']

    # --- 4. Create Spatial and Temporal Feature Matrices & Get Priors ---
    logging.info("Defining feature matrices and calculating GP prior bounds...")
    # Spatial feature matrix is just the coordinates
    Ds_features = coords
    
    # Temporal feature matrix from the MultiIndex columns.
    # This results in an array where columns are [is_weekend, hour].
    Dt_features = np.array(obs_matrix.columns.to_list())
    
    ls_s_max = get_gp_length_scale_bound(Ds_features, 'Ds')
    ls_t_max = get_gp_length_scale_bound(Dt_features, 'Dt')

    # --- 5. Construct the Design Matrix (X) ---
    logging.info("Constructing the design matrix X...")
    # To ensure the X matrix rows align with the flattened y vector, we must
    # sort the original dataframe in the same column-major order used by `make_y_Vs_Vt`.
    # `make_y_Vs_Vt` flattens by columns, which are (is_weekend, hour).
    df_sorted = df.sort_values(['is_weekend', 'hour', 'grid_y', 'grid_x'])

    # Use one-hot encoding on the time_id to create the design matrix X.
    # This automatically handles the differentiation of weekday and weekend hours.
    df_sorted['time_id'] = df_sorted['hour'] + 24 * df_sorted['is_weekend']
    X_dummies = pd.get_dummies(df_sorted['time_id'], drop_first=True, prefix='time')
    
    # --- CRITICAL FIX for Identifiability ---
    # Remove the intercept from the fixed effects design matrix X.
    # This forces the model to use the spatial and temporal random effects to
    # explain the baseline ride counts, preventing the fixed effects from
    # dominating and masking the spatial variation.
    X = X_dummies
    # Explicitly convert to a numeric dtype to avoid 'object' type arrays
    X = X.astype(np.float64)

    # --- 6. Verify Shapes and Run the Model ---
    logging.info("Verification of matrix shapes:")
    logging.info(f"  Observation Matrix (locations x time): {obs_matrix.shape}")
    logging.info(f"  Coordinates (locations x 2): {coords.shape}")
    logging.info(f"  y (obs x 1): {y.shape}")
    logging.info(f"  X (obs x features): {X.shape}")
    logging.info(f"  Vs (obs x locations): {Vs.shape}")
    logging.info(f"  Vt (obs x time): {Vt.shape}")
    logging.info(f"  Ds_features (locations x 2): {Ds_features.shape}")
    logging.info(f"  Dt_features (time x 2): {Dt_features.shape}")

    if y.shape[0] != X.shape[0]:
        logging.error("Shape mismatch between y and X. Aborting model run.")
        return

    logging.info("Starting ZINB-GP model run...")
    # These are example hyperparameters. You should tune them for your needs.
    # We pass the feature matrices to the model now.
    model_results = ZINB_GP(
        X=X.values, 
        y=y,
        coords=coords,
        Vs=Vs,
        Vt=Vt,
        Ds_features=Ds_features,
        Dt_features=Dt_features,
        priors={
            'ltPrior': {'max': ls_t_max},
            'lsPrior': {'max': ls_s_max}
        },
        # The ZINB_GP model will be adapted to not need these pre-computed matrices
        Ds=None, # No longer needed
        Dt=None, # No longer needed
        nsim=20,
        burn=10,
        print_progress=True
    )
    
    # --- 7. Process Results ---
    summarize_and_save_results(model_results, X.columns.to_list(), "model_results.npz")

def summarize_and_save_results(results: dict, x_cols: list, output_path: str):
    """
    Saves model results to a compressed .npz file and prints summary statistics
    for key hyperparameters.

    Args:
        results (dict): The dictionary of posterior samples from the model.
        output_path (str): The path to save the .npz file.
        x_cols (list): The column names of the design matrix X.
    """
    logging.info(f"Saving model results to {output_path}...")
    results['X_cols'] = np.array(x_cols, dtype=object) # Save X column names
    np.savez_compressed(output_path, **results)
    logging.info(f"Results saved successfully to {os.path.abspath(output_path)}")

    logging.info("--- Posterior Summary Statistics for Key Hyperparameters ---")
    
    # Define which scalar parameters to summarize
    params_to_summarize = [
        'Sigma1s', 'Noise1s', 'Sigma2s', 'Noise2s',
        'Sigma1t', 'Noise1t', 'Sigma2t', 'Noise2t', 'R',
    ]

    for param in params_to_summarize:
        if param in results:
            samples = results[param]
            mean = np.mean(samples)
            std = np.std(samples)
            quantiles = np.quantile(samples, [0.025, 0.5, 0.975])
            logging.info(
                f"  {param:<10}: Mean={mean:8.4f}, StdDev={std:8.4f}, "
                f"95% CI=({quantiles[0]:8.4f}, {quantiles[2]:8.4f})"
            )

    # Special handling for vector parameters like length scales
    for param_prefix in ['L1s', 'L2s', 'L1t', 'L2t']:
        if param_prefix in results:
            samples = results[param_prefix]
            for i in range(samples.shape[1]):
                mean = np.mean(samples[:, i])
                logging.info(f"  {param_prefix}[{i}]   : Mean={mean:8.4f}")

if __name__ == '__main__':
    TAXI_DATA_CSV_PATH = 'nyc_taxi_zinb_ready.csv'
    
    run_model_with_taxi_data(TAXI_DATA_CSV_PATH)
