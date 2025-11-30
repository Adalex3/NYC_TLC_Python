import pandas as pd
import numpy as np
import logging
import os
from src.model_module.utils import make_y_Vs_Vt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sigmoid(eta):
    """
    Numerically stable sigmoid function.
    """
    eta = np.minimum(700, eta)
    val = 1 / (1 + np.exp(-eta))
    return np.maximum(1e-6, np.minimum(1 - 1e-6, val))

def predict_scenario(
    coords: np.ndarray,
    time_features: np.ndarray,
    posterior_means: dict,
    is_weekend: int,
    hour: int
):
    """
    Generates predictions for all locations at a specific time scenario.

    Args:
        coords (np.ndarray): Array of spatial coordinates (n_locs, 2).
        time_features (np.ndarray): Array of temporal features used in the model (n_times, 2).
        posterior_means (dict): Dictionary of posterior mean values for model parameters.
        is_weekend (int): 0 for weekday, 1 for weekend.
        hour (int): The hour of the day (0-23).

    Returns:
        pd.DataFrame: A DataFrame with coordinates and predicted ride counts.
    """
    n_locs = coords.shape[0]
    logging.info(f"Generating predictions for {n_locs} locations for is_weekend={is_weekend}, hour={hour}...")

    # --- 1. Reconstruct the Design Matrix (X) for this specific time ---
    # This must match the structure from run_taxi_data_model.py
    time_id = hour + 24 * is_weekend
    time_col_name = f"time_{time_id}"
    
    # Create a base DataFrame for the X matrix for all locations
    # This must match the structure from run_taxi_data_model.py (no intercept)
    X_pred = pd.DataFrame(0, index=np.arange(n_locs), columns=posterior_means['X_cols'].tolist())
    if time_col_name in X_pred.columns:
        X_pred[time_col_name] = 1.0

    # --- 2. Identify the Temporal Random Effect Index ---
    # Find the index corresponding to our time scenario in the original temporal features
    time_scenario = np.array([is_weekend, hour])
    time_matches = np.all(time_features == time_scenario, axis=1)
    
    if not np.any(time_matches):
        raise ValueError(f"Time scenario (weekend={is_weekend}, hour={hour}) not found in model's temporal features.")
    
    # The model dropped the first time point, so we subtract 1 from the index
    time_idx = np.where(time_matches)[0][0] - 1

    # --- 3. Calculate Linear Predictors (eta1, eta2) ---
    # Get posterior means of parameters
    alpha, beta = posterior_means['Alpha'], posterior_means['Beta']
    a, c = posterior_means['A'], posterior_means['C']
    b, d = posterior_means['B'], posterior_means['D']
    r = posterior_means['R']

    # Reconstruct the full spatial random effects. The model uses a sum-to-zero
    # constraint by dropping the first location's effect. We prepend 0 to match n_locs.
    a_full = np.insert(a, 0, 0)
    c_full = np.insert(c, 0, 0)

    # --- CRITICAL FIX ---
    # The model was trained without an intercept in the fixed effects design matrix X
    # to ensure identifiability between fixed and random effects. The random effects
    # (a, b, c, d) capture the baseline levels. Therefore, we must also EXCLUDE the
    # fixed effects term (X @ alpha, X @ beta) during prediction.
    eta1 = a_full + b[time_idx]
    eta2 = c_full + d[time_idx]

    # --- 4. Calculate Expected Ride Count ---
    # E[Y] = P(Y > 0) * E[Y | Y > 0]
    # P(Y > 0) = pi = sigmoid(eta1)
    # E[Y | Y > 0] = mu = r * exp(eta2)
    pi = sigmoid(eta1)
    mu = r * np.exp(eta2)
    expected_y = pi * mu

    # --- 5. Format Output ---
    df_pred = pd.DataFrame({
        'grid_x': coords[:, 1],
        'grid_y': coords[:, 0],
        'predicted_rides': expected_y
    })

    return df_pred

def main():
    """
    Main function to load model results and generate predictions for specific scenarios.
    """
    # --- Load Data ---
    model_results_path = 'model_results.npz'
    raw_data_path = 'nyc_taxi_zinb_ready.csv'

    if not os.path.exists(model_results_path):
        logging.error(f"Model results not found at '{model_results_path}'. Please run the model first.")
        return

    logging.info(f"Loading model results from {model_results_path}...")
    results = np.load(model_results_path, allow_pickle=True)
    
    # Calculate posterior means for all parameters
    posterior_means = {
        key: (np.mean(results[key], axis=0) if key != 'X_cols' else results[key])
        for key in results.files
    }

    # Load original data to get coordinate and time structures
    df = pd.read_csv(raw_data_path)
    obs_matrix = df.pivot_table(index=['grid_y', 'grid_x'], columns=['is_weekend', 'hour'], values='ride_count').fillna(0)
    coords = np.array(obs_matrix.index.to_list())
    time_features = np.array(obs_matrix.columns.to_list())
    
    # Store original X column names for reconstructing the design matrix
    # This is now loaded directly from the results file.

    # --- Generate and Save Predictions ---
    # Scenario 1: Weekday Evening Peak (5 PM)
    df_weekday_peak = predict_scenario(coords, time_features, posterior_means, is_weekend=0, hour=17)
    weekday_output_path = "predictions_weekday_peak_5pm.csv"
    df_weekday_peak.to_csv(weekday_output_path, index=False)
    logging.info(f"Weekday peak predictions saved to {weekday_output_path}")

    # Scenario 2: Weekend Afternoon Peak (1 PM)
    df_weekend_peak = predict_scenario(coords, time_features, posterior_means, is_weekend=1, hour=13)
    weekend_output_path = "predictions_weekend_peak_1pm.csv"
    df_weekend_peak.to_csv(weekend_output_path, index=False)
    logging.info(f"Weekend peak predictions saved to {weekend_output_path}")

if __name__ == '__main__':
    main()