
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import logging

from src.model_module.core import ZINB_GP
from src.model_module.utils import make_y_Vs_Vt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    # Create a combined time identifier (0-23 for weekday, 24-47 for weekend)
    df['time_id'] = df['hour'] + 24 * df['is_weekend']

    # For consistency, sort the dataframe early
    df = df.sort_values(by=['grid_x', 'grid_y', 'time_id']).reset_index(drop=True)

    # --- 2. Create Observation Matrix and Coordinate Inputs ---
    logging.info("Pivoting data to create observation matrix...")
    # Create a unique location identifier for each grid cell
    df['location_id'] = df.groupby(['grid_x', 'grid_y']).ngroup()
    
    # Pivot to get (locations x time) matrix for ride counts
    # Fill missing values with 0, assuming no record means no rides
    obs_matrix = df.pivot_table(
        index='location_id', 
        columns='time_id', 
        values='ride_count'
    ).fillna(0)
    
    # Get the unique coordinates corresponding to the location_id index
    coords_df = df[['location_id', 'grid_x', 'grid_y']].drop_duplicates().sort_values('location_id')
    coords = coords_df[['grid_x', 'grid_y']].values

    # --- 3. Generate y, Vs, and Vt using the helper function ---
    logging.info("Generating y, Vs, and Vt from observation matrix...")
    data_struct = make_y_Vs_Vt(obs_matrix.values)
    y, Vs, Vt = data_struct['y'], data_struct['Vs'], data_struct['Vt']

    # --- 4. Compute Spatial and Temporal Distance Matrices ---
    logging.info("Computing distance matrices...")
    # Spatial distance matrix (Ds)
    Ds = squareform(pdist(coords, metric='euclidean'))
    # Scale up distances to improve numerical stability, as recommended by the error message.
    Ds = Ds * 1000.0
    # Temporal distance matrix (Dt)
    time_points = obs_matrix.columns.values.reshape(-1, 1)
    Dt = squareform(pdist(time_points, metric='euclidean'))

    # --- 5. Construct the Design Matrix (X) ---
    logging.info("Constructing the design matrix X...")
    # To ensure the X matrix rows align with the flattened y vector, we must
    # sort the original dataframe in the same column-major order used by `make_y_Vs_Vt`.
    # `make_y_Vs_Vt` flattens the obs_matrix column by column (i.e., by 'time_id').
    df_sorted = df.sort_values(['time_id', 'location_id'])

    # Use one-hot encoding on the time_id to create the design matrix X.
    # This automatically handles the differentiation of weekday and weekend hours.
    X_dummies = pd.get_dummies(df_sorted['time_id'], drop_first=True, prefix='time')
    
    # Add an intercept term
    X = X_dummies.copy()
    X['intercept'] = 1.0
    
    # Reorder columns to have the intercept first
    cols = ['intercept'] + [col for col in X.columns if col != 'intercept']
    X = X[cols]

    # --- 6. Verify Shapes and Run the Model ---
    logging.info("Verification of matrix shapes:")
    logging.info(f"  Observation Matrix (locations x time): {obs_matrix.shape}")
    logging.info(f"  Coordinates (locations x 2): {coords.shape}")
    logging.info(f"  y (obs x 1): {y.shape}")
    logging.info(f"  X (obs x features): {X.shape}")
    logging.info(f"  Vs (obs x locations): {Vs.shape}")
    logging.info(f"  Vt (obs x time): {Vt.shape}")
    logging.info(f"  Ds (locations x locations): {Ds.shape}")
    logging.info(f"  Dt (time x time): {Dt.shape}")

    if y.shape[0] != X.shape[0]:
        logging.error("Shape mismatch between y and X. Aborting model run.")
        return

    logging.info("Starting ZINB-GP model run...")
    # These are example hyperparameters. You should tune them for your needs.
    model_results = ZINB_GP(
        X=X.values, 
        y=y,
        coords=coords,
        Vs=Vs,
        Vt=Vt,
        Ds=Ds,
        Dt=Dt,
        nsim=2000,
        burn=1000,
        print_progress=True
    )

    # --- 7. Process Results ---
    logging.info("Model run completed.")
    # You can now save, plot, or analyze the results.
    # For example, printing the summary of posterior samples:
    # print(model_results['summary'])
    # np.save("model_results.npy", model_results)


if __name__ == '__main__':
    # !!! IMPORTANT !!!
    # Replace this with the actual path to your CSV file.
    TAXI_DATA_CSV_PATH = 'nyc_taxi_zinb_ready.csv'
    
    run_model_with_taxi_data(TAXI_DATA_CSV_PATH)
